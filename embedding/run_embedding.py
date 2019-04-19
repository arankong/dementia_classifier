from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import GroupKFold
from dementia_classifier.settings import SQL_DBANK_TEXT_FEATURES, SQL_DBANK_DIAGNOSIS, SQL_DBANK_ACOUSTIC_FEATURES, SQL_DBANK_TEXT_EMBEDDINGS, SQL_DBANK_ACOUSTIC_EMBEDDINGS
from dementia_classifier.feature_extraction.feature_sets.feature_set_list import task_specific_features
# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()

ALZHEIMERS     = ["PossibleAD", "ProbableAD"]
CONTROL        = ["Control"]


def get_data(diagnosis=ALZHEIMERS + CONTROL):
	text = pd.read_sql_table(SQL_DBANK_TEXT_FEATURES, cnx)
	acoustic = pd.read_sql_table(SQL_DBANK_ACOUSTIC_FEATURES, cnx)
	diag = pd.read_sql_table(SQL_DBANK_DIAGNOSIS, cnx)

	diag = diag[diag['diagnosis'].isin(diagnosis)]

	fv_text = pd.merge(diag, text)
	fv_acoustic = pd.merge(diag, acoustic)

	fv_text = fv_text.sample(frac=1, random_state=20)
	fv_acoustic = fv_acoustic.sample(frac=1, random_state=20)

    # Collect Labels
	label_text = [label[:3] for label in fv_text['interview']]
	label_acoustic = [label[:3] for label in fv_acoustic['interview']]
	assert label_text == label_acoustic
	labels = label_acoustic

	interviews = fv_acoustic['interview']

	drop = ['level_0', 'interview', 'diagnosis', 'gender', 'index', 'gender_int']
	drop_task_specific = drop + task_specific_features()
	
	fv_text = fv_text.drop(drop_task_specific, axis=1, errors='ignore')
	fv_text = fv_text.apply(pd.to_numeric, errors='ignore')

	fv_acoustic = fv_acoustic.drop(drop, axis=1, errors='ignore')
	fv_acoustic = fv_acoustic.apply(pd.to_numeric, errors='ignore')

	fv_text.index = labels
	fv_acoustic.index = labels
	interviews.index = labels

	group_kfold_text = GroupKFold(n_splits=10).split(fv_text, interviews, groups=labels)
	data_text = []
	for train_index, test_index in group_kfold_text:
		fold = {}
		fold["text_train"] = fv_text.values[train_index]
		fold["ids_train"] = interviews.values[train_index]
		fold["text_test"]  = fv_text.values[test_index]
		fold["ids_test"]  = interviews.values[test_index]
		data_text.append(fold)

	group_kfold_acoustic = GroupKFold(n_splits=10).split(fv_acoustic, interviews, groups=labels)
	data_acoustic = []
	for train_index, test_index in group_kfold_acoustic:
		fold = {}
		fold["acoustic_train"] = fv_acoustic.values[train_index]
		fold["ids_train"] = interviews.values[train_index]
		fold["acoustic_test"]  = fv_acoustic.values[test_index]
		fold["ids_test"]  = interviews.values[test_index]
		data_acoustic.append(fold)

	return data_text, data_acoustic 


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, input_dim, embedded_size):
        super(EmbeddingLayer, self).__init__()
        self.linear = nn.Linear(input_dim, embedded_size)
    
    def forward(self, x):
        return self.linear(x)


# Save embeddings on test fold as features
def embed_test_fold(text_encoder, acoustic_encoder, text_test, acoustic_test, interviews):
	length = len(text_test)
	text_features_embedded = np.zeros((length, 50))
	acoustic_features_embedded = np.zeros((length, 50))

	for i in range(length):
		ti = Variable(torch.from_numpy(text_test[i]).float())
		ai = Variable(torch.from_numpy(acoustic_test[i]).float())

		ti_embedded = text_encoder(ti)
		ai_embedded = acoustic_encoder(ai)

		text_features_embedded[i, :] = ti_embedded.detach().numpy()
		acoustic_features_embedded[i, :] = ai_embedded.detach().numpy()

	interview = np.reshape(interviews, (length,-1))
	text_nparray = np.hstack((interview, text_features_embedded))
	acoustic_nparray = np.hstack((interview, acoustic_features_embedded))

	return text_nparray, acoustic_nparray


def train(text_lr, acoustic_lr, num_epochs):
	text_features, acoustic_features = get_data()
	text_nparrays = np.empty((0, 51))
	acoustic_nparrays = np.empty((0, 51))

	for idx in range(10):
		print "Training for fold ", idx, " start!"
		text_fold = text_features[idx]
		text_train, text_test = text_fold["text_train"], text_fold["text_test"]
		text_interviews = text_fold["ids_test"]

		acoustic_fold = acoustic_features[idx]
		acoustic_train, acoustic_test = acoustic_fold["acoustic_train"], acoustic_fold["acoustic_test"]
		acoustic_interviews = acoustic_fold["ids_test"]

		assert text_interviews.all() == acoustic_interviews.all()
		interviews = text_interviews

		criterion = nn.HingeEmbeddingLoss(margin=0.2)	
		text_encoder = EmbeddingLayer(99, 50)
		acoustic_encoder = EmbeddingLayer(172, 50)
		text_optimizer = torch.optim.Adam(text_encoder.parameters(), lr=text_lr) 
		audio_optimizer = torch.optim.Adam(acoustic_encoder.parameters(), lr=acoustic_lr) 

		y = torch.FloatTensor([-1])
		length = len(text_train)

		for epoch in range(num_epochs):
			avg_loss = 0
			for i in range(length):
				loss_i = 0
				ti = Variable(torch.from_numpy(text_train[i]).float())
				ai = Variable(torch.from_numpy(acoustic_train[i]).float())

				for j in range(length):
					if j == i:
						continue

					tj = Variable(torch.from_numpy(text_train[j]).float())
					aj = Variable(torch.from_numpy(acoustic_train[j]).float())

					ti_embedded = text_encoder(ti).view(1,50,1)
					tj_embedded = text_encoder(tj).view(1,50,1)

					ai_embedded = acoustic_encoder(ai).view(1,50,1)
					aj_embedded = acoustic_encoder(aj).view(1,50,1)

					loss = criterion(F.cosine_similarity(ti_embedded, ai_embedded)-F.cosine_similarity(ti_embedded, aj_embedded),y) + criterion(F.cosine_similarity(ti_embedded, ai_embedded)-F.cosine_similarity(tj_embedded, ai_embedded),y)
					# loss_i += loss.data[0]
					loss_i += loss.item()
					
					text_optimizer.zero_grad()
					audio_optimizer.zero_grad()
					loss.backward()
					text_optimizer.step()
					audio_optimizer.step()

				avg_loss += (loss_i/(length-1))
				if (i+1) % 40 == 0:
					print 'The average loss for id#%d to id#%d is %.6f'%(i-39,i,avg_loss/40)
					avg_loss = 0

		print "Training for fold ", idx, " finish!"
		text_nparray, acoustic_nparray = embed_test_fold(text_encoder, acoustic_encoder, text_test, acoustic_test, interviews)

		text_nparrays = np.vstack((text_nparrays, text_nparray))
		acoustic_nparrays = np.vstack((acoustic_nparrays, acoustic_nparray))
		# torch.save(text_encoder.state_dict(), root  + 'text_embedding_param_wo_spatial.pkl')
		# torch.save(acoustic_encoder.state_dict(), root + 'acoustic_embedding_param_wo_spatial.pkl')
	
	print "Saving embeddings on test fold as features..."
	text_frame = pd.DataFrame(text_nparrays)
	text_rename = {0: "interview"}
	for i in range(1, 51):
		text_rename[i] = "t%d" % i
	text_frame.rename(columns=text_rename, inplace=True)

	acoustic_frame = pd.DataFrame(acoustic_nparrays)
	acoustic_rename = {0: "interview"}
	for i in range(1, 51):
		acoustic_rename[i] = "a%d" % i
	acoustic_frame.rename(columns=acoustic_rename, inplace=True)

	text_frame.to_sql(SQL_DBANK_TEXT_EMBEDDINGS, cnx, if_exists='replace', index=False)
	acoustic_frame.to_sql(SQL_DBANK_ACOUSTIC_EMBEDDINGS, cnx, if_exists='replace', index=False)

def save_embedding_features():
	num_epochs = 1
	text_lr = 0.01
	acoustic_lr = 0.01

	train(text_lr, acoustic_lr, num_epochs)


from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import GroupKFold
from dementia_classifier.settings import SQL_DBANK_TEXT_FEATURES, SQL_DBANK_DIAGNOSIS, SQL_DBANK_ACOUSTIC_FEATURES
from dementia_classifier.feature_extraction.feature_sets.feature_set_list import task_specific_features
# --------MySql---------
from dementia_classifier import db
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
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


def get_data_frames(diag, text_encoder, acoustic_encoder, text, acoustic, interviews):
	length = len(text)
	text_features_embedded = np.zeros((length, 50))
	acoustic_features_embedded = np.zeros((length, 50))
	for i in range(length):
		ti = Variable(torch.from_numpy(text[i]).float())
		ai = Variable(torch.from_numpy(acoustic[i]).float())

		ti_embedded = text_encoder(ti)
		ai_embedded = acoustic_encoder(ai)

		text_features_embedded[i, :] = ti_embedded.detach().numpy()
		acoustic_features_embedded[i, :] = ai_embedded.detach().numpy()

	interviews = np.reshape(interviews, (length,-1))
	text_nparray = np.hstack((interviews, text_features_embedded))
	acoustic_nparray = np.hstack((interviews, acoustic_features_embedded))

	text_frame = pd.DataFrame(text_nparray)
	text_rename = {0: "interview"}
	for i in range(1, 51):
		text_rename[i] = "t%d" % i
	text_frame.rename(columns=text_rename, inplace=True)

	acoustic_frame = pd.DataFrame(acoustic_nparray)
	acoustic_rename = {0: "interview"}
	for i in range(1, 51):
		acoustic_rename[i] = "a%d" % i
	acoustic_frame.rename(columns=acoustic_rename, inplace=True)

	return text_frame, acoustic_frame


def evaluate(text_encoder, acoustic_encoder, text_train, acoustic_train, text_test, acoustic_test, interviews_train, interviews_test):
	diagnosis=ALZHEIMERS + CONTROL
	diag = pd.read_sql_table(SQL_DBANK_DIAGNOSIS, cnx)
	diag = diag[diag['diagnosis'].isin(diagnosis)]

	text_frame_train, acoustic_frame_train = get_data_frames(diag, text_encoder, acoustic_encoder, text_train, acoustic_train, interviews_train)
	text_frame_test, acoustic_frame_test = get_data_frames(diag, text_encoder, acoustic_encoder, text_test, acoustic_test, interviews_test)

	text_frame_train, acoustic_frame_train = pd.merge(text_frame_train, diag, on=['interview']), pd.merge(acoustic_frame_train, diag, on=['interview'])
	text_frame_test, acoustic_frame_test = pd.merge(text_frame_test, diag, on=['interview']), pd.merge(acoustic_frame_test, diag, on=['interview'])

	y_text = ~text_frame_train.diagnosis.isin(CONTROL)
	y_acoustic = ~acoustic_frame_train.diagnosis.isin(CONTROL)
	assert y_text.all() == y_acoustic.all()
	y_train = y_text

	ytest_text = ~text_frame_test.diagnosis.isin(CONTROL)
	ytest_acoustic = ~acoustic_frame_test.diagnosis.isin(CONTROL)
	assert ytest_text.all() == ytest_acoustic.all()
	y_test = ytest_text

	text_frame_train, acoustic_frame_train = text_frame_train.drop(['diagnosis'], axis=1, errors='ignore'), acoustic_frame_train.drop(['diagnosis'], axis=1, errors='ignore')
	text_frame_test, acoustic_frame_test = text_frame_test.drop(['diagnosis'], axis=1, errors='ignore'), acoustic_frame_test.drop(['diagnosis'], axis=1, errors='ignore')
	combined_frame_train = pd.merge(text_frame_train, acoustic_frame_train, on=['interview'])
	combined_frame_test = pd.merge(text_frame_test, acoustic_frame_test, on=['interview'])

	combined_features_train = combined_frame_train.drop(['interview'], axis=1, errors='ignore')
	combined_features_train = combined_features_train.apply(pd.to_numeric, errors='ignore')
	combined_features_test = combined_frame_test.drop(['interview'], axis=1, errors='ignore')
	combined_features_test = combined_features_test.apply(pd.to_numeric, errors='ignore')

	text_features_train, text_features_test = text_frame_train.drop(['interview'], axis=1, errors='ignore'), text_frame_test.drop(['interview'], axis=1, errors='ignore')
	text_features_train, text_features_test = text_features_train.apply(pd.to_numeric, errors='ignore'), text_features_test.apply(pd.to_numeric, errors='ignore')

	acoustic_features_train, acoustic_features_test = acoustic_frame_train.drop(['interview'], axis=1, errors='ignore'), acoustic_frame_test.drop(['interview'], axis=1, errors='ignore')
	acoustic_features_train, acoustic_features_test = acoustic_features_train.apply(pd.to_numeric, errors='ignore'), acoustic_features_test.apply(pd.to_numeric, errors='ignore')

	model_text = LogisticRegression(max_iter=1000)
	model_acoustic = LogisticRegression(max_iter=1000)
	model_combined = LogisticRegression(max_iter=1000)

	model_text = model_text.fit(text_features_train, y_train)
	model_acoustic = model_acoustic.fit(acoustic_features_train, y_train)
	model_combined = model_combined.fit(combined_features_train, y_train)

	# Predict
	yhat_text = model_text.predict(text_features_test)
	yhat_acoustic = model_acoustic.predict(acoustic_features_test)
	yhat = model_combined.predict(combined_features_test)

	acc, fms = accuracy_score(y_test, yhat), f1_score(y_test, yhat)
	acc_text, fms_text = accuracy_score(y_test, yhat_text), f1_score(y_test, yhat_text)
	acc_acoustic, fms_acoustic = accuracy_score(y_test, yhat_acoustic), f1_score(y_test, yhat_acoustic)
	return acc_text, fms_text, acc_acoustic, fms_acoustic, acc, fms


def train(text_features, acoustic_features, text_lr, acoustic_lr, num_epochs):
	for idx in range(10):
		print "Training for fold ", idx, " start!"
		text_fold = text_features[idx]
		text_train, text_test = text_fold["text_train"], text_fold["text_test"]
		text_interviews = text_fold["ids_test"]

		acoustic_fold = acoustic_features[idx]
		acoustic_train, acoustic_test = acoustic_fold["acoustic_train"], acoustic_fold["acoustic_test"]
		acoustic_interviews = acoustic_fold["ids_test"]

		assert text_fold["ids_train"].all() == acoustic_fold["ids_train"].all()
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
		torch.save(text_encoder.state_dict(), 'embedding/text_embedding_param_{}.pkl'.format(idx))
		torch.save(acoustic_encoder.state_dict(), 'embedding/acoustic_embedding_param_{}.pkl'.format(idx))


def test(text_features, acoustic_features):
	acc_scores_text = []
	f1_scores_text = []
	acc_scores_acoustic = []
	f1_scores_acoustic = []
	acc_scores_combined = []
	f1_scores_combined = []

	for idx in range(10):
		text_fold = text_features[idx]
		text_train, text_test = text_fold["text_train"], text_fold["text_test"]
		acoustic_fold = acoustic_features[idx]
		acoustic_train, acoustic_test = acoustic_fold["acoustic_train"], acoustic_fold["acoustic_test"]

		text_embedding = EmbeddingLayer(99, 50)
		acoustic_embedding = EmbeddingLayer(172, 50)
		text_embedding.load_state_dict(torch.load('embedding/text_embedding_param_{}.pkl'.format(idx)))
		acoustic_embedding.load_state_dict(torch.load('embedding/acoustic_embedding_param_{}.pkl'.format(idx)))

		acc_text, fms_text, acc_acoustic, fms_acoustic, acc, fms = evaluate(text_embedding, acoustic_embedding, text_train, acoustic_train, text_test, acoustic_test, text_fold["ids_train"], text_fold["ids_test"])
		print "acc_text, fms_text, acc_acoustic, fms_acoustic, acc_combined, fms_combined for fold ", idx, " are ", acc_text, fms_text, acc_acoustic, fms_acoustic, acc, fms
		acc_scores_text.append(acc_text)
		f1_scores_text.append(fms_text)
		acc_scores_acoustic.append(acc_acoustic)
		f1_scores_acoustic.append(fms_acoustic)
		acc_scores_combined.append(acc)
		f1_scores_combined.append(fms)

	print "Accuracy for Embedded_L is ", np.nanmean(acc_scores_text, axis=0)
	print "F score for Embedded_L is ", np.nanmean(f1_scores_text, axis=0)
	print "Accuracy for Embedded_A is ", np.nanmean(acc_scores_acoustic, axis=0)
	print "F score for Embedded_A is ", np.nanmean(f1_scores_acoustic, axis=0)
	print "Accuracy for Embedded_L&A is ", np.nanmean(acc_scores_combined, axis=0)
	print "F score for Embedded_L&A is ", np.nanmean(f1_scores_combined, axis=0)


def embedding_experiment():
	num_epochs = 3
	text_lr = 0.01
	acoustic_lr = 0.01
	text_features, acoustic_features = get_data()

	train(text_features, acoustic_features, text_lr, acoustic_lr, num_epochs)
	test(text_features, acoustic_features)

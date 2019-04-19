import numpy as np
import pandas as pd
from dementia_classifier.analysis import util
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


class DementiaCV(object):
    """
    DementiaCV performs 10-fold group cross validation, where data points with a given label
    are confined in a single fold. This object is written with the intention of being a 
    superclass to the DomainAdaptation and BlogAdaptation subclasses, where the subclasses 
    only need to override the 'get_data_folds' method
    """

    def __init__(self, model, X=None, y=None, labels=None, silent=False):
        super(DementiaCV, self).__init__()
        self.model = model

        self.X = X
        self.y = y
        self.labels = labels
        self.columns = X.columns

        self.methods = ['default']
        self.nfolds = 10

        # Results
        self.silent = silent
        self.results    = {}
        self.best_score = {}
        self.best_k     = {}

        self.myprint("Model %s" % model)
        self.myprint("===========================")

    def get_data_folds(self, fold_type='default'):
        X, y, labels = self.X, self.y, self.labels
        if X is None or y is None:
            raise ValueError("X or y is None")

        group_kfold = GroupKFold(n_splits=self.nfolds).split(X, y, groups=labels)
        data = []
        for train_index, test_index in group_kfold:
            fold = {}
            fold["X_train"] = X.values[train_index]
            fold["y_train"] = y.values[train_index]
            fold["X_test"]  = X.values[test_index]
            fold["y_test"]  = y.values[test_index]
            fold["train_labels"]  = np.array(labels)[train_index]
            data.append(fold)
        return data

    def myprint(self, msg):
        if not self.silent:
            print msg

    def train_model(self, method='default', k_range=None, model=None):
        if model is None:
            model = self.model

        acc = []
        fms = []
        roc = []

        for idx, fold in enumerate(self.get_data_folds(method)):
            self.myprint("Processing fold: %i" % idx)

            X_train, y_train = fold["X_train"], fold["y_train"].ravel()  # Ravel flattens a (n,1) array into (n, )
            X_test, y_test   = fold["X_test"], fold["y_test"].ravel()

            acc_scores = []
            fms_scores = []
            if y_test.all():
                print "All values in y_test are the same in fold %i, ROC not defined." % idx
            roc_scores = []

            nfeats = X_train.shape[1]
            feats = util.get_top_pearson_features(X_train, y_train, nfeats)
            if k_range is None:
                k_range = range(1, nfeats+1)
            if k_range[0] == 0:
                raise ValueError("k_range cannot start with 0")
            for k in k_range:
                indices = feats[:k]
                # Select k features
                X_train_fs = X_train[:, indices]
                X_test_fs  = X_test[:, indices]

                model = model.fit(X_train_fs, y_train)

                # Predict
                yhat_probs = model.predict_proba(X_test_fs)
                yhat = model.predict(X_test_fs)

                # Save
                acc_scores.append(accuracy_score(y_test, yhat))
                fms_scores.append(f1_score(y_test, yhat))
                if y_test.all():
                    roc_scores.append(np.nan)
                else:
                    roc_scores.append(roc_auc_score(y_test, yhat_probs[:, 1]))
            # ----- save fold -----
            acc.append(acc_scores)
            fms.append(fms_scores)
            roc.append(roc_scores)

        self.results[method] = {"acc": np.asarray(acc),
                                "fms": np.asarray(fms),
                                "roc": np.asarray(roc)
                                }

        self.best_k[method]  = {"acc": np.array(k_range)[np.argmax(np.nanmean(acc, axis=0))],
                                "fms": np.array(k_range)[np.argmax(np.nanmean(fms, axis=0))],
                                "roc": np.array(k_range)[np.argmax(np.nanmean(roc, axis=0))],
                                "k_range": k_range}

        self.best_score[method] = {"acc": np.max(np.nanmean(acc, axis=0)),
                                   "fms": np.max(np.nanmean(fms, axis=0)),
                                   "roc": np.max(np.nanmean(roc, axis=0))
                                   }

        return self



class BlogCV(DementiaCV):
    """BlogCV is a subclass of DementiaCV which performs a 9-fold cross validation 
    where the test fold has contains posts from blogs not in the training fold.
    """

    def __init__(self, model, X, y, labels, silent=False, random_state=1):
        super(BlogCV, self).__init__(model, X=X, y=y, labels=labels, silent=silent)
        self.methods = ['model', 'majority_class']
        
    def get_data_folds(self, fold_type='default'):

        X, y, labels = self.X, self.y, self.labels

        testset1 = ["creatingmemories", "journeywithdementia"]
        testset2 = ["creatingmemories", "earlyonset"]
        testset3 = ["creatingmemories", "helpparentsagewell"]

        testset4 = ["living-with-alzhiemers", "journeywithdementia"]
        testset5 = ["living-with-alzhiemers", "earlyonset"]
        testset6 = ["living-with-alzhiemers", "helpparentsagewell"]

        testset7 = ["parkblog-silverfox", "journeywithdementia"]
        testset8 = ["parkblog-silverfox", "earlyonset"]
        testset9 = ["parkblog-silverfox", "helpparentsagewell"]

        folds = [testset1, testset2, testset3, testset4, testset5, testset6, testset7, testset8, testset9]
        data = []
        for fold in folds:
            train_index = ~X.index.isin(fold)
            test_index = X.index.isin(fold)
            fold = {}
            fold["X_train"] = X.values[train_index]
            fold["y_train"] = y.values[train_index]
            fold["X_test"]  = X.values[test_index]
            fold["y_test"]  = y.values[test_index]
            fold["train_labels"]  = np.array(labels)[train_index]
            data.append(fold)

        return data

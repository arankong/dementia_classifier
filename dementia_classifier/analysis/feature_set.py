import pandas as pd
from sqlalchemy import types
from cross_validators import DementiaCV
import util
import models
import data_handler
from dementia_classifier.feature_extraction.feature_sets import feature_set_list

# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------

# =================================================================
# ----------------------Save results to sql------------------------
# =================================================================
def save_selected_feature_results_to_sql(selected_feature_sets, polynomial_terms=None):
    
    full_feature_set = models.FEATURE_SETS
    new_feature_set = models.NEW_FEATURE_SETS
    new_feature_set.append('none')

    classifiers  = models.CLASSIFIERS

    prefix = "results_selected_features"

    unselected_feature_sets = [f for f in full_feature_set if f not in selected_feature_sets]

    if "halves" in selected_feature_sets:
        polynomial_terms = feature_set_list.halves_features()
    else:
        polynomial_terms = None

    to_drop = []

    for feature_set in unselected_feature_sets:
        if feature_set == "cfg":
            to_drop += feature_set_list.cfg_features()
        elif feature_set == "syntactic_complexity":
            to_drop += feature_set_list.syntactic_complexity_features()
        elif feature_set == "psycholinguistic":
            to_drop += feature_set_list.psycholinguistic_features()
        elif feature_set == "vocabulary_richness":
            to_drop += feature_set_list.vocabulary_richness_features()
        elif feature_set == "repetitiveness":
            to_drop += feature_set_list.repetitiveness_features()
        elif feature_set == "acoustics":
            to_drop += feature_set_list.acoustics_features()
        elif feature_set == "demographic":
            to_drop += feature_set_list.demographic_features()
        elif feature_set == "parts_of_speech":
            to_drop += feature_set_list.parts_of_speech_features()
        elif feature_set == "information_content":
            to_drop += feature_set_list.information_content_features()
        elif feature_set == "strips":
            to_drop += feature_set_list.strips_features()
        elif feature_set == "halves":
            to_drop += feature_set_list.halves_features()
        elif feature_set == "quadrant":
            to_drop += feature_set_list.quadrant_features()

    # to_drop += feature_set_list.coherence_score()
    for feature_set in new_feature_set:
        print 'Saving new feature: %s' % feature_set
        X, y, labels = data_handler.get_data(drop_features=to_drop, polynomial_terms=polynomial_terms)
        print "Number of features used: ",len(X.values[0])
        trained_models = {model: DementiaCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}

        save_models_to_sql_helper(trained_models, prefix)

def save_models_to_sql_helper(trained_models, prefix, if_exists='replace'):
    method = 'default'
    dfs = []
    for model in trained_models:
        cv = trained_models[model]
        k_range = cv.best_k[method]['k_range']
        for metric in models.METRICS:
            if metric in cv.results[method].keys():
                results = cv.results[method][metric]
                df = pd.DataFrame(results, columns=k_range)
                df['metric'] = metric.decode('utf-8', 'ignore')
                df['model'] = model
                dfs += [df]


    df = pd.concat(dfs, axis=0, ignore_index=True)
    typedict = {col_name: types.Float(precision=5, asdecimal=True) for col_name in df}
    typedict['metric'] = types.NVARCHAR(length=255)
    typedict['model']  = types.NVARCHAR(length=255)
    df.to_sql(prefix, cnx, if_exists=if_exists, dtype=typedict)


# =================================================================
# ----------------------Get results from sql-----------------------
# =================================================================


def get_selected_feature_results(model, metric):
    name = "results_selected_features"
    nfs = pd.read_sql_table(name, cnx, index_col='index')
    nfs = nfs[(nfs.metric == metric) & (nfs.model == model)].dropna(axis=1)
    max_nfs_k = nfs.mean().argmax()
    nfs = nfs[max_nfs_k].to_frame().reset_index(drop=True)
    nfs.columns = ['folds']

    nfs.columns = ['folds']
    nfs['model'] = model
    nfs['metric'] = metric

    return nfs

# =================================================================
# ------------------------- Get results ----------------------------
# ================================================================

def selected_feature_set_result(metric='acc', show=False):
    print "Plotting new_feature_set_plot, metric: %s" % metric
    classifiers = list(models.CLASSIFIER_KEYS)

    for fs in ['none']:
        for classifier in classifiers:
            df = get_selected_feature_results(classifier, metric)
            util.print_ci_from_df(df['folds'], fs, classifier)

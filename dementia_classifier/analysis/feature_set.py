import pandas as pd
from sqlalchemy import types
import util
import models
from util import bar_plot

# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------

# =================================================================
# ----------------------Save results to sql------------------------
# =================================================================

def save_models_to_sql_helper(trained_models, ablation_set, prefix, if_exists='replace'):
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

    name = "%s_%s" % (prefix, ablation_set)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    typedict = {col_name: types.Float(precision=5, asdecimal=True) for col_name in df}
    typedict['metric'] = types.NVARCHAR(length=255)
    typedict['model']  = types.NVARCHAR(length=255)
    df.to_sql(name, cnx, if_exists=if_exists, dtype=typedict)


# =================================================================
# ----------------------Get results from sql-----------------------
# =================================================================


def get_new_feature_results(new_feature_set, model, metric, absolute=True, poly=True):
    reference = "results_new_features_none"
    ref = pd.read_sql_table(reference, cnx, index_col='index')
    ref = ref[(ref.metric == metric) & (ref.model == model)].dropna(axis=1)

    max_ref_k = ref.mean().argmax()
    ref = ref[max_ref_k].to_frame().reset_index(drop=True)
    ref.columns = ['folds']

    # nfs == new feature set
    if new_feature_set == 'halves' and poly:
        name = "results_new_features_poly_%s" % new_feature_set
    else:
        name = "results_new_features_%s" % new_feature_set
    nfs = pd.read_sql_table(name, cnx, index_col='index')
    nfs = nfs[(nfs.metric == metric) & (nfs.model == model)].dropna(axis=1)
    max_nfs_k = nfs.mean().argmax()
    nfs = nfs[max_nfs_k].to_frame().reset_index(drop=True)
    nfs.columns = ['folds']

    if not absolute:
        nfs = nfs - ref

    nfs.columns = ['folds']
    nfs['model'] = model
    nfs['metric'] = metric
    nfs['new_feature_set'] = new_feature_set

    return nfs

# =================================================================
# ------------------------- Make plots ----------------------------
# ================================================================

def new_feature_set_plot(metric='acc', absolute=True, poly=True, show=False):
    print "Plotting new_feature_set_plot, metric: %s" % metric
    classifiers = list(models.CLASSIFIER_KEYS)
    new_features = []
    if absolute:
        new_features += ['none']
    new_features += models.NEW_FEATURE_SETS
    classifiers.remove('DummyClassifier')
    dfs = []

    for fs in new_features:
        for classifier in classifiers:
            df = get_new_feature_results(fs, classifier, metric, absolute=absolute, poly=poly)
            util.print_ci_from_df(df['folds'], fs, classifier)
            dfs.append(df)

    dfs = pd.concat(dfs)
    dfs = dfs.replace('none', 'baseline')

    y_lim = (.68, .90)

    if metric == 'acc':
        y_label = "Accuracy"
    elif metric == 'fms':
        y_label = "F-Measure"
    else:
        y_label = "AUC"
        y_lim = (.70, .95)

    figname = 'new_feature_plot_%s' % metric
    title = 'Performance w/ New Feature Sets'
    if not absolute:
        y_label = "Change in %s" % y_label
        y_lim = (-.10, .10)
        figname = figname + '_relative'
        title = 'Change in Performance w/ New Feature Sets'

    plot_specs = {
        'x_col': 'new_feature_set',
        'y_col': 'folds',
        'hue_col': 'model',
        'x_label': 'Feature Set',
        'y_label': y_label,
        'y_lim': y_lim,
        'figsize': (10, 8),
        'fontsize': 20,
        'font_scale': 1.2,
        'labelsize': 15,
        'show': show,
        'title': title,
    }
    
    # We use polynomial terms as well for halves
    if poly:
        dfs = dfs.replace('halves', 'halves+quadratic')
    else:
        figname = figname + '_without_quadratic'
    
    figname = figname + '.pdf'
    bar_plot(dfs, figname, **plot_specs)

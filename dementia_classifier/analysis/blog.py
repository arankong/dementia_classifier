import pandas as pd
from sqlalchemy import types
from cross_validators import BlogCV
from dementia_classifier.analysis import data_handler
from dementia_classifier.settings import BLOG_RESULTS, SQL_BLOG_SUFFIX
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


def save_blog_results_to_sql():
    X, y, labels = data_handler.get_blog_data()
    classifiers = models.CLASSIFIERS
    trained_models = {model: BlogCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}
    save_blogs_to_sql_helper(trained_models, if_exists='append')


def save_blogs_to_sql_helper(trained_models, if_exists='replace'):
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

    df.to_sql(BLOG_RESULTS, cnx, if_exists=if_exists, dtype=typedict)


# =================================================================
# ----------------------Get results from sql-----------------------
# =================================================================


def get_blog_results(model, metric):
    name = "results_blog"
    table = pd.read_sql_table(name, cnx, index_col='index')
    table = table[(table.metric == metric) & (table.model == model)].dropna(axis=1)
    max_k = table.mean().argmax()
    df = table[max_k].to_frame()
    df.columns = ['folds']
    df['model'] = model
    df['metric'] = metric
    return df


def get_blog_feature(feature):
    dfs = []
    for blog in models.BLOG_NAMES:
        name = "%s_%s" % (blog, SQL_BLOG_SUFFIX)
        table = pd.read_sql_table(name, cnx)
        df = pd.DataFrame(table[feature].astype(float))
        df['blog'] = blog
        dfs.append(df)

    return pd.concat(dfs)
# =================================================================
# ------------------------- Make plots ----------------------------
# =================================================================

def blog_plot():
    print "Plotting blog_plot"
    metrics = models.METRICS
    dfs = []
    for classifier in models.CLASSIFIER_KEYS:
        for metric in metrics:
            df = get_blog_results(classifier, metric)
            util.print_ci_from_df(df['folds'], classifier, metric)
            dfs.append(df)

    dfs = pd.concat(dfs)
    
    plot_specs = {
        'x_col': 'model',
        'y_col': 'folds',
        'hue_col': 'metric',
        'x_label': 'Model',
        'y_label': 'Performance',
        'font_scale': 1.2,
        'fontsize': 20,
        'rotation': 15
    }

    figname = 'blog_plot.pdf'

    bar_plot(dfs, figname, **plot_specs)

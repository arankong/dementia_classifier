import pandas as pd
import numpy as np
from pandas.io import sql
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
from dementia_classifier.feature_extraction.feature_sets import feature_set_list
from dementia_classifier.analysis import data_handler

from dementia_classifier.settings import PLOT_PATH

# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------

sns.set_style('whitegrid')

def bar_plot(dfs, figname, **kwargs):
    x_col      = kwargs.pop('x_col', None)
    y_col      = kwargs.pop('y_col', None)
    hue_col    = kwargs.pop('hue_col', None)
    x_label    = kwargs.pop('x_label', "")
    y_label    = kwargs.pop('y_label', "")
    fontsize   = kwargs.pop('fontsize', 10)
    titlesize  = kwargs.pop('titlesize', 16)
    y_lim      = kwargs.pop('y_lim', (0, 1))
    show       = kwargs.pop('show', False)
    order      = kwargs.pop('order', None)
    dodge      = kwargs.pop('dodge', True)
    figsize    = kwargs.pop('figsize', (10, 8))
    font_scale = kwargs.pop('font_scale', 0.8)
    labelsize  = kwargs.pop('labelsize', None)
    title      = kwargs.pop('title', "")
    errwidth   = kwargs.pop('errwidth', 0.75)
    rotation   = kwargs.pop('rotation', None)
    capsize    = kwargs.pop('capsize', 0.2)
    tight_layout    = kwargs.pop('tight_layout', False)
    # legend_loc    = kwargs.pop('legend_loc', None)

    if x_col is None:
        raise ValueError("No x_column entered")

    if y_col is None:
        raise ValueError("No y_column entered")

    sns.set_style('whitegrid')
    sns.set(font_scale=font_scale)
    plt.figure(figsize=figsize)
    if rotation is not None:
        plt.xticks(rotation=rotation)

    ax = sns.barplot(x=x_col, y=y_col, hue=hue_col, data=dfs, order=order, palette=colormap(), dodge=dodge, ci=90, errwidth=errwidth, capsize=capsize)
    fig = ax.get_figure()
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(labelsize=labelsize)
    ax.legend(prop={'size': 7})

    if 'model' in dfs.columns:
        name_fix = {
            'DummyClassifier': "MajorityClass",
            'SVC': "SVM",
            'KNeighbors': "KNN",
        }
        labels = [name_fix[item.get_text()] if item.get_text() in name_fix else item.get_text() for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        for t in ax.legend().texts:
            if t.get_text() in name_fix:
                t.set_text(name_fix[t.get_text()])

    # Fix metric labels hack
    if 'metric' in dfs.columns and len(dfs.metric.unique()) == 3:
        new_labels = ["Accuracy", "AUC", "F-Measure"]
        for t, l in zip(ax.legend().texts, new_labels):
            t.set_text(l)

    fig.suptitle(title, y=.93, fontsize=titlesize)
    if tight_layout:
        fig.tight_layout()
    if y_lim is not None:
        plt.ylim(*y_lim)
    if show:
        plt.show()
    else:
        fig.savefig(PLOT_PATH + figname, bbox_inches="tight")


def print_ci_from_df(df, model, metric):
    fm_ci = st.t.interval(0.90, len(df) - 1, loc=np.mean(df), scale=st.sem(df))
    mean = df.mean()
    print '%s: (%s of %0.3f, and 90%% CI=%0.3f-%0.3f)' % (model, metric, mean, fm_ci[0], fm_ci[1])
    # return mean, fm_ci[0], fm_ci[1]


def new_features_dataset_helper(key, polynomial_terms=True):
    to_drop = feature_set_list.new_features()
    if key == "strips":
        new_feature_set = feature_set_list.strips_features()
        to_drop = [f for f in to_drop if f not in new_feature_set]
    elif key == "halves":
        new_feature_set = feature_set_list.halves_features()
        to_drop = [f for f in to_drop if f not in new_feature_set]
    elif key == "quadrant":
        new_feature_set = feature_set_list.quadrant_features()
        to_drop = [f for f in to_drop if f not in new_feature_set]
    elif key == "discourse":
        new_feature_set = feature_set_list.discourse_features()
        to_drop = [f for f in to_drop if f not in new_feature_set]
        new_feature_set = []
    elif key == "none":
        return data_handler.get_data(drop_features=to_drop)
    else:
        raise ValueError("Incorrect key")

    if polynomial_terms:
        return data_handler.get_data(drop_features=to_drop, polynomial_terms=new_feature_set)
    else:
        return data_handler.get_data(drop_features=to_drop)


def delete_sql_tables(bad_tables):
    for table in bad_tables:
        print 'Deleting %s' % table
        sql.execute('DROP TABLE IF EXISTS %s' % table, cnx)


def get_top_pearson_features(X, y, n, return_correlation=False):
    df = pd.DataFrame(X).apply(pd.to_numeric)
    df['y'] = y
    corr_coeff = df.corr()['y'].abs().sort_values(inplace=False, ascending=False)
    if return_correlation:
        return corr_coeff
    else:
        return corr_coeff.index.values[1:n + 1].astype(int)

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import GroupKFold
from dementia_classifier.feature_extraction.feature_sets import feature_set_list
from dementia_classifier.settings import SQL_DBANK_TEXT_FEATURES, SQL_DBANK_DIAGNOSIS, SQL_DBANK_DEMOGRAPHIC, SQL_DBANK_ACOUSTIC_FEATURES, SQL_DBANK_TEXT_EMBEDDINGS, SQL_DBANK_ACOUSTIC_EMBEDDINGS
# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------

ALZHEIMERS     = ["PossibleAD", "ProbableAD"]
CONTROL        = ["Control"]
CONTROL_BLOGS  = ["earlyonset", "helpparentsagewell", "journeywithdementia"]
DEMENTIA_BLOGS = ["creatingmemories", "living-with-alzhiemers", "parkblog-silverfox"]

# ------------------
# Diagnosis keys
# - Control
# - MCI
# - Memory
# - Other
# - PossibleAD
# - ProbableAD
# - Vascular
# ------------------

# ===================================================================================
# ----------------------------------DementiaBank-------------------------------------
# ===================================================================================


def get_data(diagnosis=ALZHEIMERS + CONTROL, drop_features=None, polynomial_terms=None):

    # Read from sql
    text = pd.read_sql_table(SQL_DBANK_TEXT_FEATURES, cnx)
    demo = pd.read_sql_table(SQL_DBANK_DEMOGRAPHIC, cnx)
    diag = pd.read_sql_table(SQL_DBANK_DIAGNOSIS, cnx)
    acoustic = pd.read_sql_table(SQL_DBANK_ACOUSTIC_FEATURES, cnx)

    text_embeddings = pd.read_sql_table(SQL_DBANK_TEXT_EMBEDDINGS, cnx)
    acoustic_embeddings = pd.read_sql_table(SQL_DBANK_ACOUSTIC_EMBEDDINGS, cnx)

    # Add diagnosis
    diag = diag[diag['diagnosis'].isin(diagnosis)]
    fv = pd.merge(text, diag)
    # Merge lexical and acoustic
    fv = pd.merge(fv, acoustic, on=['interview'])
    # Add demographics
    fv = pd.merge(fv, demo)

    fv = pd.merge(fv, text_embeddings)
    fv = pd.merge(fv, acoustic_embeddings)
    # Randomize
    fv = fv.sample(frac=1, random_state=20)

    # Collect Labels
    labels = [label[:3] for label in fv['interview']]
    # Diagnoses not in control marked receive positive label
    y = ~fv.diagnosis.isin(CONTROL)
    # Clean
    drop = ['level_0', 'interview', 'diagnosis', 'gender', 'index', 'gender_int']

    X = fv.drop(drop, axis=1, errors='ignore')

    X = X.apply(pd.to_numeric, errors='ignore')

    X.index = labels
    y.index = labels

    if drop_features:
        X = X.drop(drop_features, axis=1, errors='ignore')

    X = make_polynomial_terms(X, polynomial_terms)

    return X, y, labels


def make_polynomial_terms(data, cols):
    if cols is None:
        return data

    for f1, f2 in itertools.combinations_with_replacement(cols, 2):
        if f1 == f2:
            prefix = 'sqr_'
        else:
            prefix = 'intr_'
        data[prefix + f1 + "_" + f2] = data[f1] * data[f2]

    return data


# ===================================================================================
# ----------------------------------BlogData-----------------------------------------
# ===================================================================================


def get_blog_data(keep_only_good=True, random=20, drop_features=None):
    # Read from sql
    cutoff_date = pd.datetime(2017, 4, 4)  # April 4th 2017 was when data was collected for ACL paper

    demblogs = pd.concat([pd.read_sql_table("%s_text_features" % blog, cnx) for blog in DEMENTIA_BLOGS])
    ctlblogs = pd.concat([pd.read_sql_table("%s_text_features" % blog, cnx) for blog in CONTROL_BLOGS])
    qual     = pd.read_sql_table("blog_quality", cnx)
    
    demblogs['dementia'] = True
    ctlblogs['dementia'] = False
    
    fv = pd.concat([demblogs, ctlblogs], ignore_index=True)

    # Remove recent posts (e.g. after paper was published)
    qual['date'] = pd.to_datetime(qual.date)
    qual = qual[qual.date < cutoff_date]

    # keep only good blog posts
    if keep_only_good:
        qual = qual[qual.quality == 'good']

    demblogs = pd.merge(demblogs, qual[['id', 'blog']], on=['id', 'blog'])

    # Randomize
    fv = fv.sample(frac=1, random_state=random)

    # Get labels
    labels = fv['blog'].tolist()

    # Split
    y = fv['dementia'].astype('bool')

    # Clean
    drop = ['blog', 'dementia', 'id']
    X = fv.drop(drop, 1, errors='ignore')

    if drop_features:
        X = X.drop(drop_features, axis=1, errors='ignore')

    X = X.apply(pd.to_numeric, errors='ignore')
    X.index = labels
    y.index = labels

    return X, y, labels


def get_blog_scatterplot_data(keep_only_good=True, random=20):
    # Read from sql
    blogs = pd.read_sql_table("blogs", cnx)
    qual = pd.read_sql_table("blog_quality", cnx)
    lengths = pd.read_sql_table("blog_lengths", cnx)

    if keep_only_good:
        qual = qual[qual.quality == 'good']

    # keep only good
    data = pd.merge(blogs, qual[['id', 'blog', 'date']], on=['id', 'blog'])
    data = pd.merge(data, lengths, on=['id', 'blog'])

    # Fix reverse post issue
    data.id = data.id.str.split('_', expand=True)[1].astype(int)
    data.id = -data.id

    data.date = pd.to_datetime(data.date)

    return data


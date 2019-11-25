"""
============================
XGB modelling with embedding features - for Zalo AI challenge
============================
Author: Le Anh Tho
"""
# Import libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from utils.df_preprocessing import *
import os, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':

    # =======================
    # ==== LOAD ALL DATA ====
    # =======================
    # Load training and test data
    train_info = pd.read_csv("data/train_info.tsv", delimiter="\t")
    train_rank = pd.read_csv("data/train_rank.csv")
    test_info = pd.read_csv("data/test_info.tsv", delimiter="\t")

    # Lowercase columns
    train_info.columns = map(str.lower, train_info.columns)
    train_rank.columns = map(str.lower, train_rank.columns)
    test_info.columns = map(str.lower, test_info.columns)

    # Load audio features and merge them into the main data
    train_ft = pd.merge(pd.read_csv("audio-features/train_song_metadata.csv"), train_rank, how="left", on="id")
    train = (pd.merge(train_ft, train_info, how="left", on="id")
             .pipe(feature_pipeline)
             .sort_values(by=["title", "label"])
             .drop_duplicates(subset=["title", "artist_name", "composers_name", "release_time"])
             )   
    test = (pd.merge(test_info, pd.read_csv("audio-features/test_song_metadata.csv"), how="left", on="id")
            .pipe(feature_pipeline)
            )

    # Load embeddings
    cat_cols = [
        "artist_id",
        "composers_id",
        "album",
        "genre",
        "release_month",
        "release_year",
        "release_dow",
        "release_doy",
        "release_dom",
        "release_hour",
        "is_cover",
        "is_remix",
        "is_beat",
        "is_ost",
    ]
    for c in cat_cols:
        # Load the embedding data
        temp = pd.read_csv('embeddings/{}_embedding.csv'.format(c))
        temp = temp[temp[c] != 'undefined']
        if c in ['release_doy', 'release_hour', 'release_dom', 'release_month', 'release_year', 'release_dow']:
            temp[c] = temp[c].astype('int')
        train = pd.merge(train, temp, how="left", on=c)
        test = pd.merge(test, temp, how="left", on=c)
    test = test.fillna(0)

    # ===========================
    # ==== INPUT PREPARATION ====
    # ===========================
    # Hot features
    # hot_param = {'min_titles': 3, 'rank_': 3.18873352460533, 'max_number': 98}
    hot_params = {'min_titles': 5, 'rank_': 3.580519979374021, 'max_number': 94}
    hot_artist_list = get_hottest(train, **hot_params)
    hot_composer_list = get_hottest(train, colname='composers_name', **hot_params)
    train['hot_artist'] = train['artist_name'].apply(is_hot, args=(hot_artist_list,))
    train['hot_composer'] = train['composers_name'].apply(is_hot, args=(hot_composer_list,))
    test['hot_artist'] = test['artist_name'].apply(is_hot, args=(hot_artist_list,))
    test['hot_composer'] = test['composers_name'].apply(is_hot, args=(hot_composer_list,))
    # List of features
    features = [
        "artist_id",
        "composers_id",
        "release_year",
        "n_artists",
        "n_composers",
        "artist_is_composer",
        "word_count",
        "duration",
        "album",
        "genre",
        "hot_artist",
        "hot_composer",
        "is_cover",
        "is_remix",
        "is_beat",
        "is_ost",
        "release_hour_sin",
        "release_hour_cos",
        "release_month_sin",
        "release_month_cos",
        "release_dow_sin",
        "release_dow_cos",
        "release_doy_sin",
        "release_doy_cos",
        "release_dom_sin",
        "release_dom_cos",
    ]
    to_drop = [
        'id', 'title', 'artist_name', 'composers_name',
        'release_time',
        'release_hour_sin', 'release_date',
       'release_hour_cos', 'release_month_sin', 'release_month_cos',
       'release_dow_sin', 'release_dow_cos', 'release_doy_sin',
       'release_doy_cos', 'release_dom_sin', 'release_dom_cos',
       'weekend_release',
       'is_medley', 
    ] + cat_cols
    matrix = train.drop(to_drop + ['label'], axis=1).copy()
    test_matrix = test.drop(to_drop, axis=1).copy()

    X, y = matrix.to_numpy(), train["label"].values
    test_matrix = test_matrix.to_numpy()

    # ===================
    # ==== MODELLING ====
    # ===================
    if not os.path.exists("ensemble-models"):
        os.mkdir("ensemble-models")

    params = {
        'colsample_bytree': 0.45236665464729203, 
        'min_child_weight': 50, 
        'gamma': 1, 
        'subsample': 0.8793476883382675, 
        'max_depth': 10, 
        'n_estimators': 1906, 
        'learning_rate': 0.01936476662662972
    }
    model = xgb.XGBRegressor(n_jobs=-1, objective="reg:squarederror",silent=0, **params)
    model.fit(X, y)
    test["label"] = np.clip(model.predict(test_matrix).round(4), 1, 10)
    print("Creating CSV file...")
    test[["id", "label"]].to_csv("ensemble-models/xgb_embedding.csv", index=False, header=False)
    
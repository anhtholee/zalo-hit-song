# Import libraries
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cb
from tqdm import tqdm
# from scipy.stats import hmean
# from scipy.stats.mstats import gmean
from utils.df_preprocessing import *
from utils.models import NaiveRankEstimator
from sklearn.linear_model import Ridge
# from sklearn.svm import SVR, LinearSVC, SVC
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import warnings, os
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

    # ===========================
    # ==== INPUT PREPARATION ====
    # ===========================
    # Hot features
    # hot_param = {'min_titles': 3, 'rank_': 3.18873352460533, 'max_number': 98}
    hot_param = {'min_titles': 5, 'rank_': 3.580519979374021, 'max_number': 94}
    hot_artist_list = get_hottest(train, **hot_param)
    hot_composer_list = get_hottest(train, colname='composers_name', **hot_param)
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
    
    matrix = train[features].copy()
    test_matrix = test[features].copy()
    
    X, y = matrix.copy(), train["label"].copy()
    Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.12, random_state=2019)
    Xtest = test_matrix.copy()
    # Label encoding for categorical features
    # merged = pd.concat((matrix, test_matrix), ignore_index=True)
    for col in ['artist_id', 'composers_id', 'album', 'genre']:
        le = LabelEncoderExt()
        le.fit(X[col])
        X[col] = le.transform(X[col])
        test_matrix[col] = le.transform(test_matrix[col])

    # ===================
    # ==== MODELLING ====
    # ===================
    if not os.path.exists("ensemble-models"):
        os.mkdir("ensemble-models")

    xgb_params = {
        "colsample_bytree": 0.310346775670412,
        "gamma": 0,
        "subsample": 0.9019616530715178,
        "max_depth": 7,
        "n_estimators": 1996,
        "learning_rate": 0.04262864544598512,
    }

    cb_params = {
        'objective': 'RMSE', 
        'iterations': 1521, 
        'colsample_bylevel': 0.29108635840325625, 
        'eta': 0.04098290676797156, 
        'depth': 8, 
        'boosting_type': 'Plain', 
        'bootstrap_type': 'Bernoulli', 
        'subsample': 0.7441863450462856
    }

    models = [
        ("xgb", xgb.XGBRegressor(n_jobs=-1, objective="reg:squarederror", **xgb_params)),
        ("cb", cb.CatBoostRegressor(**cb_params)),
        ("naive", NaiveRankEstimator(agg_method="gmean", default_method="gmean")),
    ]

    # Create predictions from each model
    preds = pd.DataFrame(test['id'])
    i = 1
    for name, m in tqdm(models, total=len(models)):
        colname = str(i) + '_' + name
        if name == 'cb':
            m.fit(
                Xtrain, ytrain,
                cat_features=['artist_id', 'composers_id', 'album', 'genre'],
                eval_set=(Xval, yval),
                verbose=False,
                early_stopping_rounds=100
            )
            preds[colname] = np.clip(m.predict(Xtest).round(4), 1, 10)
        else:
            m.fit(X, y)
            preds[colname] = np.clip(m.predict(test_matrix).round(4), 1, 10)
        # print("Creating CSV file...")
        preds[["id", colname]].to_csv("ensemble-models/{}_{}.csv".format(i, name), index=False, header=False)
        i += 1
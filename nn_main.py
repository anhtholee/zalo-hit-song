"""
============================
Neural net modelling - for Zalo AI challenge
============================
Author: Le Anh Tho
"""
# Import libraries
import numpy as np
import pandas as pd
# from scipy.stats import hmean
# from scipy.stats.mstats import gmean
from sklearn.preprocessing import StandardScaler
from utils.df_preprocessing import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from embedder2 import preprocessing
from embedder2.regression import Embedder
import warnings
import random as rn
import tensorflow as tf
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['PYTHONHASHSEED'] = '42'

def get_embedding_df(embed_matrix, entity_list, colname="artist_id", prefix="a"):
    df1 = pd.DataFrame(entity_list, columns=[colname])
    df2 = pd.DataFrame(
        embed_matrix,
        columns=["{}_{}".format(prefix, i) for i in range(embed_matrix.shape[1])],
    )
    return pd.concat((df1, df2), axis=1).fillna("undefined")


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
    
    # List of features
    features = [
        "artist_id",
        "composers_id",
        "release_month",
        "release_year",
        "release_dow",
        "release_doy",
        "release_dom",
        "release_hour",
        "n_artists",
        "n_composers",
        "artist_is_composer",
        "word_count",
        "duration",
        "album",
        "genre",
        "is_cover",
        "is_remix",
        "is_beat",
        "is_ost",
    ]
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
    matrix = train[features].copy()
    test_matrix = test[features].copy()
    matrix[cat_cols] = matrix[cat_cols].astype("object")
    test_matrix[cat_cols] = test_matrix[cat_cols].astype("object")

    # Scaling
    num_ft = ["n_artists", "n_composers", "word_count", "duration"]
    # print("Numerical ft: {}".format(num_ft))
    scaled_ft = StandardScaler().fit_transform(matrix[num_ft])
    test_scaled_ft = StandardScaler().fit_transform(test_matrix[num_ft])

    matrix[num_ft] = scaled_ft
    test_matrix[num_ft] = test_scaled_ft

    # Prepare X_train & y_train
    X_train, y_train = matrix, train["label"]
    # print("Features: {}".format(X_train.columns))

    merged = pd.concat((matrix, test_matrix))
    cat_sz = preprocessing.categorize(merged)
    emb_sz = preprocessing.pick_emb_dim(
        cat_sz,
        max_dim=10,
        emb_dims=[5, 5, 6, 1, 1, 4, 1, 8, 7, 5, 1, 1, 1, 1],
        include_unseen=False,
    )
    # print("Embedding: {}".format(emb_sz))
    X_train_encoded, encoders = preprocessing.encode_categorical(X_train, X_train)
    X_test_encoded, test_encoders = preprocessing.encode_categorical(test_matrix, X_train,)
    
    print(X_train_encoded.shape)
    
    # ===================
    # ==== MODELLING ====
    # ===================
    if not os.path.exists("nn-weights"):
        os.mkdir("nn-weights")
    if not os.path.exists("embeddings"):
        os.mkdir("embeddings")
    if not os.path.exists("ensemble-models"):
        os.mkdir("ensemble-models")
    # Fit the NN model to the full training set - Fit n times to reduce variance
    embeddings = None
    n_runs = 5
    for i in range(1,n_runs+1):
        # Fit 5 times
        np.random.seed(i)
        rn.seed(i)
        tf.random.set_seed(i)   
        print("Run {}...".format(i))
        embedder = Embedder(emb_sz, loss='mean_absolute_error', hiddens=[64,32], dropout=0.1, activation='swish')
        es = EarlyStopping(monitor='val_rmse_k', mode='min', patience=4, verbose=1)
        mc = ModelCheckpoint('nn-weights/nn_12.h5', monitor='val_rmse_k', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
        embedder.fit(X_train_encoded[:], y_train[:], epochs=25, batch_size=32, verbose=1, early_stop=es, checkpoint=mc)
        
        # Get the model with the best trained weight
        best_model = Embedder(emb_sz, weight_path='nn-weights/nn_12.h5', loss='mean_absolute_error', hiddens=[64,32], dropout=0.1, activation='swish')
        best_model.fit(X_train_encoded[:], y_train[:])

        # Predict on test set
        print("Generate predictions...")
        test["run_{}".format(i)] = np.clip(best_model.predict(X_test_encoded), 1, 10)
        print("Add new embedding matrices")
        if embeddings is None:
            embeddings = best_model.get_embeddings()
        else:
            new_embeddings = best_model.get_embeddings()
            for c, e in embeddings.items():
                embeddings[c] += new_embeddings[c]

    # Averaging the runs
    test['label'] = np.average(test[["run_{}".format(i) for i in range(1, n_runs+1)]], axis=1).round(4)
    # Output csv file
    print("Creating CSV file...")
    test[["id", "label"]].to_csv("ensemble-models/nn.csv", index=False, header=False)

    # Get embedding matrices
    print("Get avg embedding matrices...")
    # embeddings = embedder.get_embeddings()
    for c in cat_cols:
        # Averaging
        embeddings[c] /= n_runs
        embed_matrix = embeddings[c]
        entity_list = encoders[c].classes_
        ft_df = get_embedding_df(embed_matrix, entity_list, colname=c, prefix=c)
        ft_df.to_csv("embeddings/{}_embedding.csv".format(c), index=False)
    
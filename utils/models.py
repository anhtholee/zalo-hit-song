"""
============================
Modelling:  Custom model(s) for Zalo AI challenge
============================
Author: Le Anh Tho
"""
import numpy as np
import pandas as pd
from scipy.stats import hmean
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
# from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import learning_curve, KFold, StratifiedKFold
# from tqdm import tqdm

class NaiveRankEstimator(BaseEstimator):
    """
    Estimate the songs' ranks using a simple, naive strategy:
    - If the artist/composer/album/genre is known --> return the aggregated rank (mean/median, etc.)
    - If neither is known --> return the year's average ranking

    Attributes
    ----------
    agg_method : str
        The method for artist/composer/album/genre aggregation. Can be 'mean', 'hmean' or 'gmean'
    default_method : str
        The method for default aggregation. Can be 'mean', 'hmean' or 'gmean'
    df : DataFrame
        The dataframe which contains the columns from X train
    artists : DataFrame
        The summary table for artists
        composers : DataFrame
        The summary table for composers
    year_aggs : DataFrame
        The summary table for release years
    is_fitted_ : bool
        sklearn's param
    """

    def __init__(self, agg_method='mean', default_method='mean', weights=[0, 0]):
        self.agg_method = agg_method
        self.default_method = default_method
        self.w1 = weights[0]
        self.w2 = weights[1]

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        # X, y = check_X_y(X, y, accept_sparse=True)
        # self._df = X.loc[:,['artist_id', 'composers_id', 'release_year']]
        self._df = X[['artist_id', 'composers_id', 'album', 'genre',
                      'release_year']].copy(deep=True)
        self._df['label'] = y.copy()
        # assert self._df.isnull().any() == False
        # print(self._df.count())
        # print(self._df.isnull().sum())
        # Get aggregations for artist and composers
        if self.agg_method == 'mean':
            self._artists = self._df.groupby('artist_id').agg(
                {'label': 'mean'}).reset_index().rename(columns={'label': 'artist_agg'})
            self._composers = self._df.groupby('composers_id').agg(
                {'label': 'mean'}).reset_index().rename(columns={'label': 'composers_agg'})
            self._albums = self._df.groupby('album').agg(
                {'label': 'mean'}).reset_index().rename(columns={'label': 'album_agg'})
            self._genres = self._df.groupby('genre').agg(
                {'label': 'mean'}).reset_index().rename(columns={'label': 'genre_agg'})
        elif self.agg_method == 'gmean':
            self._artists = self._df.groupby('artist_id').agg({'label': lambda x: gmean(
                x)}).reset_index().rename(columns={'label': 'artist_agg'})
            self._composers = self._df.groupby('composers_id').agg({'label': lambda x: gmean(
                x)}).reset_index().rename(columns={'label': 'composers_agg'})
            self._albums = self._df.groupby('album').agg({'label': lambda x: gmean(
                x)}).reset_index().rename(columns={'label': 'album_agg'})
            self._genres = self._df.groupby('genre').agg({'label': lambda x: gmean(
                x)}).reset_index().rename(columns={'label': 'genre_agg'})
        elif self.agg_method == 'hmean':
            self._artists = self._df.groupby('artist_id').agg({'label': lambda x: hmean(
                x)}).reset_index().rename(columns={'label': 'artist_agg'})
            self._composers = self._df.groupby('composers_id').agg({'label': lambda x: hmean(
                x)}).reset_index().rename(columns={'label': 'composers_agg'})
            self._albums = self._df.groupby('album').agg({'label': lambda x: hmean(
                x)}).reset_index().rename(columns={'label': 'album_agg'})
            self._genres = self._df.groupby('genre').agg({'label': lambda x: hmean(
                x)}).reset_index().rename(columns={'label': 'genre_agg'})
        else:  # random sample
            self._artists = self._df.groupby('artist_id').agg({'label': lambda x: x.sample(
                1)}).reset_index().rename(columns={'label': 'artist_agg'})
            self._composers = self._df.groupby('composers_id').agg({'label': lambda x: x.sample(
                1)}).reset_index().rename(columns={'label': 'composers_agg'})
            self._albums = self._df.groupby('album').agg({'label': lambda x: x.sample(
                1)}).reset_index().rename(columns={'label': 'album_agg'})
            self._genres = self._df.groupby('genre').agg({'label': lambda x: x.sample(
                1)}).reset_index().rename(columns={'label': 'genre_agg'})

        # Get Aggregation by year
        if self.default_method == 'mean':
            self._year_aggs = self._df.groupby('release_year').agg(
                {'label': 'mean'}).reset_index().rename(columns={'label': 'year_agg'})
        elif self.default_method == 'gmean':
            self._year_aggs = self._df.groupby('release_year').agg(
                {'label': lambda x: gmean(x)}).reset_index().rename(columns={'label': 'year_agg'})
        elif self.default_method == 'hmean':
            self._year_aggs = self._df.groupby('release_year').agg(
                {'label': lambda x: hmean(x)}).reset_index().rename(columns={'label': 'year_agg'})
        else:  # random sample
            self._year_aggs = self._df.groupby('release_year').agg(
                {'label': lambda x: x.sample(1)}).reset_index().rename(columns={'label': 'year_agg'})

        self._albums.loc[self._albums['album'] == 'unknown', 'album'] = np.nan
        # self._df.drop_duplicates(
        #     ['artist_id', 'composers_id', 'release_year'], inplace=True)
        self._df.drop(['label'], axis=1, inplace=True)
        # `fit` should always return `self`
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features). Features: artist_id, composers_id, release_year
            The testing input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of predictions
        """
        # X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        X = X[['artist_id', 'composers_id', 'album', 'genre', 'release_year', 'hot_artist', 'hot_composer']].copy(deep=True)
        X = pd.merge(X, self._albums, on=['album'], how='left')
        X = pd.merge(X, self._artists, on=['artist_id'], how='left')
        X = pd.merge(X, self._composers, on=['composers_id'], how='left')
        X = pd.merge(X, self._genres, on=['genre'], how='left')
        X = pd.merge(X, self._year_aggs, on=['release_year'], how='left')
        labels = np.where(
            X.artist_agg.notnull(),
            X.artist_agg - (self.w1 * X.hot_artist + self.w2 * X.hot_composer),
            np.where(X.composers_agg.notnull(),
                     X.composers_agg - (self.w1 * X.hot_artist + self.w2 * X.hot_composer),
                     np.where(X.album_agg.notnull(),
                            X.album_agg - (self.w1 * X.hot_artist + self.w2 * X.hot_composer),
                            np.where(X.genre_agg.notnull(), 
                                X.genre_agg - (self.w1 * X.hot_artist + self.w2 * X.hot_composer),
                                X.year_agg - (self.w1 * X.hot_artist + self.w2 * X.hot_composer) 
                            )
                            
                     )
                        
                    )
        )

        return np.clip(labels, 1, 10)


class ModelStacking(object):
    """
    2-level model stacking estimator:
    - The base models (1st level) are used to fit the training data and prediction on out-of-fold training data will be used as new features.
    - The base models are then used to predict the test data to generate the stacked test data compatible with the stacked training data
    - The meta model (2nd level) is used on the new transformed data.

    Attributes
    ----------
    k : int
        Number of folds
    meta_model : object
        The meta model (2nd level)
    models :
        List of base (1st level) models
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in
        regression).
    X_train_stacked : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed input samples, predictions from the base model
    """

    def __init__(self, base_estimators, meta_estimator, n_folds=5,):
        super(ModelStacking, self).__init__()
        self.models = base_estimators
        self.meta_model = meta_estimator
        self.k = n_folds

    def fit(self, X, y):
        """A reference implementation of a fitting function.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in
        regression).
    Returns
    -------
    self : object
        Returns self.
    """
        self._X = X.copy()
        self._y = pd.to_numeric(y.copy()).values
        # Define K-fold
        kfold = KFold(n_splits=self.k, shuffle=True, random_state=42)
        X_train_stacked = None
        print("Base models chosen: {}".format([m.__class__.__name__ for m in self.models]))
        print("Fitting base models...")
        fold = 0
        for train_index, test_index in kfold.split(self._X):
            print("Fold {}...".format(fold+1))
            X_train, X_test = self._X.iloc[train_index], self._X.iloc[test_index]
            y_train, y_test = self._y[train_index], self._y[test_index] 

            # Train the models on the corresponding fold of the training data
            # Then use the out-of-fold predictions as a new feature for each
            # model
            X_fold = None
            for i, m in enumerate(self.models):
                m.fit(X_train, y_train)
                if X_fold is not None:
                    X_fold = np.column_stack((X_fold, m.predict(X_test)))
                else:
                    X_fold = m.predict(X_test)

            if X_train_stacked is not None:
                X_train_stacked = np.vstack((X_train_stacked, X_fold))
            else:
                X_train_stacked = X_fold
            fold+=1

        self._X_train_stacked = X_train_stacked
        print("Finished building {} stacked features for training data.".format(len(self.models)))
        self.is_fitted_ = True
        return self

    def predict(self, X, y=None):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features). Features: artist_id, composers_id, release_year
            The testing input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of predictions
        """
           # X = check_array(X, accept_sparse=True)
           # Build stacked test data
        check_is_fitted(self, 'is_fitted_')
        X_test_stacked = None
        print("Building stacked test data...")
        for i, m in enumerate(self.models):
            m.fit(self._X, self._y)
            if X_test_stacked is not None:
                X_test_stacked = np.column_stack((X_test_stacked, m.predict(X)))
            else:
                X_test_stacked = m.predict(X)

        print("Fitting meta model - {}...".format(self.meta_model.__class__.__name__))
        self.meta_model.fit(self._X_train_stacked, self._y)
        return self.meta_model.predict(X_test_stacked)

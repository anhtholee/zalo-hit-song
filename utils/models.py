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

"""
============================
Data preprocessing helpers for Zalo AI challenge
============================
Author: Le Anh Tho
"""
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import uniform, randint, hmean
from scipy.stats.mstats import gmean
from functools import reduce

# Credit: https://stackoverflow.com/a/56876351/4123563
class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(
            list(data_list) + ["Unknown"])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = [
                    "Unknown" if x == unique_item else x for x in new_data_list
                ]

        return self.label_encoder.transform(new_data_list)


def time_feature_pipeline(df):
    """Create time feature columns for the data
    
    Args:
        df (DataFrame): Input data
    
    Returns:
        DataFrame: Output data
    """
    return df.assign(
        release_time=lambda df: pd.to_datetime(
            df["release_time"], infer_datetime_format=True
        ),
        release_date=lambda df: df["release_time"].dt.date,
        release_month=lambda df: df["release_time"].dt.month,
        release_year=lambda df: df["release_time"].dt.year,
        release_dow=lambda df: df["release_time"].dt.dayofweek,
        release_doy=lambda df: df["release_time"].dt.dayofyear,
        release_dom=lambda df: df["release_time"].dt.day,
        release_hour=lambda df: df["release_time"].dt.hour,
        release_hour_sin=lambda df: np.sin(
            df.release_hour * (2 * np.pi / 24)),
        release_hour_cos=lambda df: np.cos(
            df.release_hour * (2 * np.pi / 24)),
        release_month_sin=lambda df: np.sin(
            (df.release_month-1) * (2 * np.pi / 12)),
        release_month_cos=lambda df: np.cos(
            (df.release_month-1) * (2 * np.pi / 12)),
        release_dow_sin=lambda df: np.sin(df.release_dow * (2 * np.pi / 7)),
        release_dow_cos=lambda df: np.cos(df.release_dow * (2 * np.pi / 7)),
        release_doy_sin=lambda df: np.sin(
            (df.release_doy-1) * (2 * np.pi / (365 + df.release_time.dt.is_leap_year))),
        release_doy_cos=lambda df: np.cos(
            (df.release_doy-1) * (2 * np.pi / (365 + df.release_time.dt.is_leap_year))),
        release_dom_sin=lambda df: np.sin(
            (df.release_dom-1) * (2 * np.pi / df.release_time.dt.daysinmonth)),
        release_dom_cos=lambda df: np.cos(
            (df.release_dom-1) * (2 * np.pi / df.release_time.dt.daysinmonth)),
    )


def artist_composer(x):
    """Check if artist is also composer of such title
    
    Args:
        x (Series): column
    
    Returns:
        int: 1 or 0 indicating an artist is in the list of composers of that title (or vice versa) or not
    """
    return (
        1
        if len(
            set([s.strip() for s in x["artist_name"].split(",")]).intersection(
                set([s.strip() for s in x["composers_name"].split(",")])
            )
        )
        > 0
        else 0
    )


def get_hottest(df, colname='artist_name', min_titles=3, rank_=3, max_number=50):
    """Get hottest artist/composer list from the data based on number of titles, average ranking
    
    Args:
        df (DataFrame): Input data
        colname (str): column name
        min_titles (int): Minimum number of titles an artist/composer has in order to be considered 'hot'
        rank_ (float): Maximum ranking number an artist/composer has in order to be considered 'hot'
        max_number (int): Maximum number of artists/composers in the list
    
    Returns:
        list: List of hot artists/composers
    """
    n_ = 'n_artists' if colname == 'artist_name' else 'n_composers'
    df = df.copy()
    hot_df = df[df[n_] == 1].groupby(
        [colname]).agg({"label": ["count", "mean"]})
    hot_df.columns = hot_df.columns.droplevel()
    hot_list = (
        hot_df.query("count >= @min_titles and mean <= @rank_")
        .sort_values(by=["mean", "count"], ascending=[True, False])
        .head(max_number)
        .index.tolist()
    )
    if colname == 'album':
        return hot_list
    else:
        l = [s.split(",") for s in hot_list]
        return [item.strip() for sub_list in l for item in sub_list]


def is_hot(col, hot_list):
    """Check if a title features a hot artist/composer
    
    Args:
        col (Series): column
        hot_list (list): the hot arists/composers list
    
    Returns:
        int: 1 or 0 indicating a title is hot or not
    """
    if len(set([s.strip() for s in col.split(",")]).intersection(set(hot_list))) > 0:
        return 1
    else:
        return 0


def feature_pipeline(df):
    """Create feature columns for the data
    
    Args:
        df (DataFrame): Input data
    
    Returns:
        DataFrame: Output data
    """
    df.loc[df.genre == "Nhạc Đạo", "genre"] = "Nhạc Tôn Giáo"
    return (df
        .pipe(time_feature_pipeline)
        .assign(
            word_count=lambda df: df.title.str.split().str.len(),
            weekend_release=lambda df: np.where(df.release_dow.isin([5, 6]), 1, 0),
            is_cover=lambda df: np.where(
                df.title.str.lower().str.contains("cover"), 1, 0),
            is_remix=lambda df: np.where(
                df.title.str.lower().str.contains("remix"), 1, 0),
            is_beat=lambda df: np.where(
                df.title.str.lower().str.contains("beat"), 1, 0),
            is_ost=lambda df: np.where(
                df.title.str.lower().str.contains("ost"), 1, 0),
            is_medley=lambda df: np.where(
                df.title.str.lower().str.contains("liên khúc"), 1, 0),
            n_artists=lambda df: df["artist_name"].str.split(",").apply(len),
            n_composers=lambda df: df["composers_name"].str.split(",").apply(len),
            artist_is_composer=lambda df: df.apply(artist_composer, axis=1)
        )
    )

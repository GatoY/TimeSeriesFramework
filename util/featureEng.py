import pandas as pd
import datetime
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
import sys
from util_log import logger


def filling_missing_value(df):
    """filling missing value. 
    Args:
        df: a DataFrame, contains all data with cols: 'Inside', 'Outside', 'Sales'

    Returns:
        df: a DataFrame. missing data filled
    """

    # TODO Define ur filling algorithm here.

    return df


def lagging(df,
            lag_len=50,
            lag_cols=[]
            ):
    """ Generate lagging features.
    Args:
        df: a DataFrame, contains all columns in cols_lag
        len_lag: Int, the length of lagging features
        cols_lag: List of String, the columns needs to generate lagging features 

    Returns:
        df: a DataFrame with lagging features, for example 'col_lag_1'.
    """

    # lag
    for lag in range(1, lag_len):
        for col in lag_cols:
            d_lag = df.shift(periods=lag)
            df[col + '_lag_' + str(lag)] = d_lag[col]
    return df


def sliding_window(df,
                   sliding_window_len=50,
                   sliding_cols=[]
                   ):
    """ Generate sliding window features.
    Args:
        df: a DataFrame, contains all columns in cols_lag
        len_sliding_window: Int, the length of sliding window, >2 
        cols_sliding: List of String, the columns needs to generate sliding window features 

    Returns:
        df: a DataFrame with sliding window features, for example 'col_mean_2'.
    """

    def roll(data, roll_size, col):
        roll = data[col].shift(1).rolling(roll_size)
        data[col + '_mean_' + str(roll_size)] = roll.mean()
        data[col + '_std_' + str(roll_size)] = roll.std()
        data[col + '_max_' + str(roll_size)] = roll.max()
        data[col + '_min_' + str(roll_size)] = roll.min()
        return data

    for roll_size, col in zip(range(2, sliding_window_len), sliding_cols):
        df = roll(df, roll_size, col)

    df = df.dropna()
    return df


def filter_peak_period(df,
                       PREDICT_START_DATE,
                       peak_period_list=[['2017-12-08', '2018-02-08'],
                                         ['2018-12-10', '2019-02-10']]
                       ):
    """ Filter peak period.
    Args:
        df: a DataFrame
        peak_period_list: a 2-dims List, contains peak periods. Format: [[peak1_start, peak1_end], ..]

    Returns:
        df: a DataFrame without peak periods.
    """

    for [peak_start, peak_end] in peak_period_list:
        df = df[~df.dateTime.isin(
            pd.date_range(
                datetime.datetime.strptime(peak_start, '%Y-%m-%d'),
                datetime.datetime.strptime(peak_end, '%Y-%m-%d')))]
        df = df.reset_index(drop=True)
    return df


def gen_datasets(df,
                 dataset_dir,
                 PREDICT_LENGTH,
                 changing_cols=['month', 'weekday', 'Inside', 'influence']
                 ):
    """ Generate train test sets for PREDICT_LENGTH of models.
    Args:
        df: a DataFrame
        dataset_dir: String, a dir to record train test sets.

    Returns:
        train_frame_dict: a Dict: {Int: Dataframe}
    """

    train_frame_dict = {}

    for i in range(PREDICT_LENGTH):
        # i_frame: ith model
        logger.debug('for the %sth model' % i)

        t = df[changing_cols].shift(-i)
        df[changing_cols] = t[changing_cols]

        df.to_csv(dataset_dir + str(i) + '.csv', index=False)
        train_frame_dict[i] = df
    return train_frame_dict


def onehot(df):
    """ Generate onehot features 
    Args:
        df: a DataFrame

    Returns:
        month_weekday_data: Numpy array. onehot features.
    """

    month_weekday_data = OneHotEncoder(categories=[range(1, 13), range(7)]).fit_transform(
        df[['month', 'weekday']]).toarray()
    return month_weekday_data


def feature_lab(d):
    """ Get features and labels 
    Args:
        d: a DataFrame

    Returns:
        features: Series
        labels: Series
    """

    month_weekday_data = onehot(d)
    lagging_sliding_data = d.drop(
        columns=['month', 'weekday']).values
    features = np.hstack((month_weekday_data, lagging_sliding_data))
    labels = d['label']
    return features, labels


def train_valid_split(train_frame_dict,
                      start_date,
                      train_test_dir,
                      predict_length
                      ):
    """ Train valid sets split
    Args:
        df: a DataFrame

    Returns:
        month_weekday_data: Numpy array. onehot features.
    """

    train_valid_dict = {}
    for i in range(predict_length):

        df = train_frame_dict[i]
        df.datetime = pd.to_datetime(df.dateTime)
        train_data = df[df['dateTime'] < start_date]
        test_data = df[df['dateTime'] == start_date]

        train_X, train_y = feature_lab(train_data)
        valid_X, valid_y = feature_lab(test_data)

        train_valid_dict[i] = {'train_X': train_X,
                               'train_y': train_y,
                               'valid_X': valid_X,
                               'valid_y': valid_y}

        # save files npy
        np.save(train_test_dir + str(i) + '_train_X.npy', train_X)
        np.save(train_test_dir + str(i) + '_train_y.npy', train_y)
        np.save(train_test_dir + str(i) + '_valid_X.npy', valid_X)
        np.save(train_test_dir + str(i) + '_valid_y.npy', valid_y)

    return train_valid_dict

# ------- HACKED BY Yu Liu ------- #
# v2.0

import pandas as pd
import numpy as np
import datetime
import pickle
import math
import pymysql

from sklearn.preprocessing import OneHotEncoder

from util.util_log import logger
from util.featureEng import (
    lagging,
    sliding_window,
    filling_missing_value,
    filter_peak_period,
    gen_datasets,
    train_valid_split
)
from util.models import grid_search
from util.config import (

    PREDICT_START_DATE,
    PREDICT_LENGTH,

    LAG_LENGTH,
    LAG_COLS,
    SLIDING_WINDOW_LENGTH,
    SLIDING_COLS,
    MODEL_LIST,

    PREIDCT_DATA_DIR,
    PREDICT_DATASET_DIR,
    PREDICT_TMP_DIR,
    PREDICT_MODEL_DIR,
)


def fetch_data():
    """ Get Raw Data.
    Args:

    Returns:
        df: a DataFrame. Raw data
    """
    pass
    df = pd.DataFrame()
    return df


def main():

    predict_start_date = datetime.datetime.strptime(
        PREDICT_START_DATE, '%Y-%m-%d')
    predict_lag_start_date = predict_start_date - \
        datetime.timedelta(days=max(LAG_LENGTH, SLIDING_WINDOW_LENGTH))

    # Raw_data
    logger.debug('fetch data')
    sql_data = fetch_data()

    # filling missing value
    df = filling_missing_value(df)

    # lag features
    df = lagging(df,
                 lag_len=LAG_LENGTH,
                 lag_cols=LAG_COLS)
    # sliding features
    df = sliding_window(df,
                        sliding_window_len=SLIDING_WINDOW_LENGTH,
                        sliding_cols=SLIDING_COLS
                        )

    logger.debug('sort and clean')
    # sort and clean
    df = df.dropna()
    df = df.sort_values(by=['storeId', 'dateTime'])
    df = df.drop_duplicates(subset=['storeId', 'dateTime'], keep='first')
    df = df.reset_index(drop=True)

    # record intermedia file
    df.to_csv(PREDICT_DATASET_DIR + 'predict_dataset.csv', index=False)


    # load pre trained models
    model_list = []
    for i in range(PREDICT_LENGTH):
        model_list.append(pickle.load(
            open(PREDICT_MODEL_DIR + 'best_model_' + str(i) + '.pkl', 'rb')))

    # predict
    store_list = future_frame.storeId.unique()
    for i in range(PREDICT_LENGTH):
        features = get_feature(df)
        label = model_list[i].predict(features)[0]
        

if __name__ == '__main__':
    main()

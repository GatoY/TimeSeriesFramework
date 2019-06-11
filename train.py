# ------- HACKED BY Yu Liu ------- #

import pymysql
import pandas as pd
import os
import datetime
import math

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

    # Valid period, Use the closest period of predict length to tune models.  [VALID_START_DATE, VALID_END_DATE)
    VALID_END_DATE = datetime.datetime.strptime(PREDICT_START_DATE, '%Y-%m-%d')
    VALID_START_DATE = VALID_END_DATE-datetime.timedelta(days=PREDICT_LENGTH)

    logger.debug('Get data')
    # fetch raw data
    df = fetch_data()
    df.to_csv(PREDICT_DATASET_DIR + 'raw_data.csv', index=False)

    logger.debug('Filling missing value, lag, sliding window')

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

    logger.debug('filter peak periods')
    # filter peak periods
    df = filter_peak_period(df, PREDICT_START_DATE)

    logger.debug('sort and clean')
    # sort and clean
    df = df.dropna()
    df = df.sort_values(by=['storeId', 'dateTime'])
    df = df.drop_duplicates(subset=['storeId', 'dateTime'], keep='first')
    df = df.reset_index(drop=True)

    # record intermedia file
    df.to_csv(PREDICT_DATASET_DIR + 'datasets.csv', index=False)

    logger.debug('gen train datasets')
    # gen train datasets
    train_frame_dict = gen_datasets(
        df,
        PREDICT_DATASET_DIR,
        PREDICT_LENGTH,
        changing_cols=[])

    logger.debug('train valid split')
    # train valid split
    train_valid_dict = train_valid_split(
        train_frame_dict,
        VALID_START_DATE,
        PREDICT_TMP_DIR,
        PREDICT_LENGTH)

    logger.debug('grid search training')

    # grid search
    grid_search(
        VALID_START_DATE,
        PREDICT_LENGTH,
        PREDICT_TMP_DIR,
        PREDICT_MODEL_DIR,
        train_valid_dict,
        model_list=MODEL_LIST,
        verbose=2
    )


if __name__ == '__main__':
    main()

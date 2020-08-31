"""
module that supplies functions that augment, enhance, or expedite pandas and its use cases.
"""

import pandas as pd


def descriptive_stats(df: pd.DataFrame, percentiles=[.25, .5, .75], exclude=None, datetime_is_numeric=False):
    """
    The pandas.DataFrame.describe() function omits datatype, missing value count, and median value per column when
    generating descriptive statistics for a DataFrame. These are important to get a general overview of a dataset. This
    descriptive_stats() function generates those and the stats already provided in pandas.DataFrame.describe() for a
    DataFrame passed in as an input parameter. All other parameters are default describe() parameters.

    :param df: pandas.DataFrame for which to generate descriptive statistics
    :return: a pandas.DataFrame containing the descriptive stats, similar to pandas.DataFrame.describe()
    """
    dstats_df = pd.concat(
        [
            df.dtypes,
            df.isna().sum(),
            df.median(numeric_only=True)
        ],
        axis=1
    )
    dstats_df = dstats_df.rename(columns={
        0: "type",
        1: "missing",
        2: "median"
    })
    dstats_df = dstats_df.T
    dstats_df = pd.concat([dstats_df, df.describe(percentiles, include='all', exclude=exclude)])  #TODO: descibe might return a Series, not a df
    return dstats_df


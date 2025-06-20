import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def random_sampling(data_df: pd.DataFrame, sample_size=1000) ->pd.DataFrame:
    """
    select randomly samples from a dataframe
    :param data_df: dataframe to sample
    :param sample_size: sample size
    :return: return sample dataframe
    """
    # Check input arguments
    assert data_df.shape[0] > sample_size, 'Sample size must be smaller than or equal to number of data points'
    # Generate synthetic data
    random_sample_df = data_df.sample(n=sample_size, random_state=42)

    # Perform K-S evaluation
    for column in random_sample_df.columns:
        ks_stat, p_value = ks_2samp(data_df[column], random_sample_df[column])
        print(f"Random Sampling column '{column}': KS Statistic = {ks_stat}, p-value = {p_value}")

    return random_sample_df



from preprocess.sampling import random_sampling
import pandas as pd


def test_random_sampling():
    cars_train_preprocessed_df = pd.read_csv('../data/preprocess/cars_train_preprocessed.csv')
    sample_size = 20000 # 20K
    cars_train_preprocessed_sample_df = random_sampling(cars_train_preprocessed_df, sample_size=sample_size)

    # Save sample dataframe
    sample_data_filepath = '../data/preprocess/cars_train_preprocessed_sample_df.csv'
    cars_train_preprocessed_sample_df.to_csv(sample_data_filepath, index=False)

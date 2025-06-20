from train.train_autoencoder import train_autoencoder


def test_train_autoencoder():
    cars_preprocessed_filepath = r'../data/preprocess/cars_train_preprocessed.csv'
    #cars_preprocessed_filepath = r'../data/preprocess/cars_train_preprocessed_sample_df.csv'
    batch_size = 128
    train_autoencoder(cars_preprocessed_filepath, batch_size=batch_size)

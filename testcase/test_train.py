from train.train import train_autoencoder


def test_train_autoencoder():
    cars_preprocessed_filepath = r'../data/preprocess/cars_preprocessed.csv'
    train_autoencoder(cars_preprocessed_filepath)

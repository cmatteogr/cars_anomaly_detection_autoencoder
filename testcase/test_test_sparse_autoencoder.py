from test_m.test_sparse_autoencoder import test as test_model


def test_train_autoencoder():
    model_filepath = '../artifacts/anomaly_detection_sparce_kl_autoencoder_model_local.pth'
    properties_data_filepath = '../data/preprocess/cars_test_preprocessed.csv'

    test_model(model_filepath, properties_data_filepath)

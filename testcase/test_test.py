from evaluation.evaluate_autoencoder import test as test_model


def test_train_autoencoder():
    model_filepath = '../artifacts/anomaly_detection_model.pth'
    properties_data_filepath = '../data/preprocess/cars_test_preprocessed.csv'

    test_model(model_filepath, properties_data_filepath)

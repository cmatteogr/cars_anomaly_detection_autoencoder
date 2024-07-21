from preprocess.preprocess import preprocess


def test_preprocess():
    cars_filepath = r'../data/preprocess/cars.csv'
    train_inputer = True
    isolation_forest_contamination = 0.15
    cars_o_df, cars_df, preprocess_data = preprocess(cars_filepath, train_inputer=train_inputer,
                                                     isolation_forest_contamination=isolation_forest_contamination)

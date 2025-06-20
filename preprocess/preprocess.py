# import project libraries
from sklearnex import patch_sklearn

patch_sklearn()

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction import FeatureHasher
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
import re
import json
import os

from constants import RELEVANT_PREPROCESS_COLUMNS, ARTIFACTS_FOLDER_PATH, DATA_FOLDER_PATH


# Apply msrp value
def map_msrp(msrp):
    """
    Replace 0 values by null
    :param msrp: manufacturer's suggested retail price
    """
    if msrp == 0:
        return np.nan
    return msrp


def clean_exterior_color(exterior_color):
    """
    Clean exterior color feature
    :param exterior_color: text to clean
    :return: text cleaned
    """
    # Check if value is empty
    if pd.isna(exterior_color):
        return 'unknown'
    # Convert interior_color to lower case
    exterior_color = exterior_color.lower()
    # Remove special characters
    exterior_color = re.sub(r'[\W_+w/\/]', ' ', exterior_color)
    # Remove double spaces
    exterior_color = re.sub(r'\s+', ' ', exterior_color)
    # Apply trim 
    exterior_color = exterior_color.strip()
    # Return formated text
    return exterior_color


def get_exterior_color_phrase_vector(exterior_color_phrase, model):
    """
    transform exterior color text to vector
    :param exterior_color_phrase: text to transform
    :param model: model word to vector
    :return: text vector
    """
    exterior_color_words = exterior_color_phrase.split()
    exterior_color_word_vectors = [model.wv[word] for word in exterior_color_words if word in model.wv]
    if not exterior_color_word_vectors:
        print(f"No words found in model for phrase: {exterior_color_phrase}")
        return np.nan
    return sum(exterior_color_word_vectors) / len(exterior_color_word_vectors)


def clean_interior_color(interior_color):
    """
    Clean interior color feature
    :param interior_color: text to clean
    :return: text cleaned
    """
    # Check if value is empty
    if pd.isna(interior_color):
        return 'unknown'
    # Convert interior_color to lower case
    interior_color = interior_color.lower()
    # Remove special characters
    interior_color = re.sub(r'[\W_+w/\/]', ' ', interior_color)
    # Remove double spaces
    interior_color = re.sub(r'\s+', ' ', interior_color)
    # Return formated text
    return interior_color


def get_interior_color_phrase_vector(interior_color_phrase, model):
    """
    transform interior color text to vector
    :param interior_color_phrase: text to transform
    :param model: model word to vector
    :return: text vector
    """
    interior_color_words = interior_color_phrase.split()
    interior_color_word_vectors = [model.wv[word] for word in interior_color_words if word in model.wv]
    if not interior_color_word_vectors:
        print(f"No words found in model for phrase: {interior_color_phrase}")
        return np.nan
    return sum(interior_color_word_vectors) / len(interior_color_word_vectors)


def map_drivetrain(drivetrain):
    """
    Group the drive train by categories
    :param drivetrain: Car drive train
    :return: Grouped drive train
    """
    if pd.isna(drivetrain):
        return np.nan
    # Apply lower case and replace special characters
    drivetrain = str(drivetrain).lower().replace('-', ' ')

    # NOTE: this feature may be an open vocabulary, so another strategy may be considered
    match drivetrain:
        case 'all wheel drive' | 'four wheel drive' | 'awd' | '4wd' | '4x2' | 'all wheel drive with locking and limited slip differential' | '4matic':
            return 'All-wheel Drive'
        case 'rear wheel drive' | 'rwd':
            return 'Rear-wheel Drive'
        case 'front wheel drive' | 'fwd' | 'front wheel drive' | '2wd':
            return 'Front-wheel Drive'
        case 'unknown':
            return np.nan
        case _:
            raise Exception(f"No expected drive train: {drivetrain}")


def clean_cat(cat):
    """
    Clean cat feature
    :param cat: text to clean
    :return: text cleaned
    """
    # Check if value is empty
    if pd.isna(cat):
        return 'unknown'
    # Convert cat to lower case
    cat = cat.lower()
    # Split by '_' and join again by ' '
    cat = ' '.join(cat.split('_'))
    # Remove double spaces
    cat = re.sub(r'\s+', ' ', cat)
    # Return formated text
    return cat


# Calculate the vectors feature avegare
def get_cat_phrase_vector(cat_phrase, model):
    """
    transform cat_phrase text to vector
    :param cat_phrase: text to transform
    :param model: model word to vector
    :return: text vector
    """
    cat_words = cat_phrase.split()
    cat_word_vectors = [model.wv[word] for word in cat_words if word in model.wv]
    if not cat_word_vectors:
        print(f"No words found in model for phrase: {cat_phrase}")
        return np.nan
    return sum(cat_word_vectors) / len(cat_word_vectors)


def map_fuel_type(fuel_type):
    """
    Group by fuel types
    :param fuel_type: Car fuel type
    :return: Fuel type category
    """
    if pd.isna(fuel_type):
        return np.nan

    # NOTE: this feature may be an open vocabulary, so another strategy may be considered
    match fuel_type:
        case 'Gasoline' | 'Gasoline Fuel' | 'Diesel' | 'Premium Unleaded' | 'Regular Unleaded' | 'Premium Unleaded' | 'Diesel Fuel':
            return 'Gasoline'
        case 'Electric' | 'Electric with Ga':
            return 'Electric'
        case 'Hybrid' | 'Plug-In Hybrid' | 'Plug-in Gas/Elec' | 'Gas/Electric Hyb' | 'Hybrid Fuel' | 'Bio Diesel' | 'Gasoline/Mild Electric Hybrid' | 'Natural Gas' | 'Electric and Gas Hybrid' | 'Gasoline/Mild Electric Hy':
            return 'Hybrid'
        case 'Flexible Fuel' | 'E85 Flex Fuel' | 'Flexible' | 'Flex Fuel Capability':
            return 'Flexible'
        case _:
            print(f"No expected fuel type: {fuel_type}")
            return np.nan


def map_stock_type(stock_type):
    """
    Map stock_type to binary value
    :param stock_type: stock type New/Used
    :return: Binary stock_type
    """
    if pd.isna(stock_type):
        return np.nan

    match stock_type:
        case 'New':
            return True
        case 'Used':
            return False
        case _:
            raise Exception(f"No expected stock type: {stock_type}")


def preprocess(cars_filepath, test_size=0.2, price_threshold=1500, make_frequency_threshold=300,
               model_hash_batch_size=20, exterior_color_vector_size=5, interior_color_vector_size=5,
               cat_vector_size=3, train_inputer=False, isolation_forest_contamination=0.1):
    """
    Preprocess cars data
    :param cars_filepath: Cars datasource filepath
    :param test_size: Test size to split dataset
    :param price_threshold: Price min value threshold
    :param make_frequency_threshold: Make category min frequency value
    :param model_hash_batch_size: Model hash batch size
    :param exterior_color_vector_size: exterior_color vector size
    :param interior_color_vector_size: interior_color vector size
    :param cat_vector_size: cat vector size
    :param train_inputer: Indicates whether imputer model is trained or not
    :param isolation_forest_contamination: Outlier removal contamination
    :return: cars_df, y_train, X_test, y_test, preprocess_config_data
    """
    print("Star preprocess")

    print("####### Collect data")
    # Read CSV file
    print("Read data from data source")
    cars_df = pd.read_csv(cars_filepath, index_col=0)
    # Assign index
    cars_df.index = cars_df['listing_id']

    print("####### Clean data")
    # Remove duplicates
    print("Remove duplicates")
    cars_df.drop_duplicates(subset='listing_id', inplace=True)
    # Copy the original data
    cars_o_df = cars_df.copy()
    print(f"Dataframe shape after remove duplicates: {cars_df.shape}")
    # Filter relevant features
    print("Remove irrelevant features")
    cars_df = cars_df[RELEVANT_PREPROCESS_COLUMNS]
    # Remove NaN target
    print("Remove rows with empty target")
    cars_df = cars_df.loc[~cars_df['price'].isna()]
    # Remove cars with price under threshold
    print(f"Remove rows under target threshold: {price_threshold}")
    cars_df = cars_df.loc[cars_df['price'] >= price_threshold]
    # Remove NaN drivetrain
    print("Remove rows with empty drivetrain")
    cars_df = cars_df.loc[~cars_df['drivetrain'].isna()]
    # Remove NaN fuel_type
    print("Remove rows with empty fuel_type")
    cars_df = cars_df[~cars_df['fuel_type'].isna()]

    # Remove make values with low count
    print("Remove make values with low count")
    # Define the threshold for category frequency
    # Compute the frequency of each category
    make_category_counts = cars_df['make'].value_counts()
    # Identify categories that exceed the threshold
    make_valid_categories = make_category_counts[make_category_counts > make_frequency_threshold].index
    # Filter the DataFrame to exclude rows with these categories
    cars_df = cars_df[cars_df['make'].isin(make_valid_categories)]

    print(f"Data set shape: {cars_df.shape}")

    print("####### Transform data")
    # ### Apply Features transformation
    # Apply msrp transformation
    print("Apply msrp transformation")
    cars_df['msrp'] = cars_df['msrp'].map(map_msrp)

    # Apply model transformation
    print("Apply model transformation")
    train_model_data = cars_df['model'].apply(lambda x: {x: 1}).tolist()
    # Define the number of hash space
    n_hash = int(len(cars_df['model'].unique()) / model_hash_batch_size)  # This values is a hyperparameter
    # Initialize FeatureHasher
    hasher_model_model = FeatureHasher(n_features=n_hash, input_type='dict')
    # Apply FeatureHasher
    train_model_hashed = hasher_model_model.transform(train_model_data)
    # Generate model hashed dataframe
    train_model_hashed_df = pd.DataFrame(train_model_hashed.toarray(),
                                         columns=[f'model_hashed_{i}' for i in range(train_model_hashed.shape[1])],
                                         index=cars_df.index)
    # Concatenate the dataframes
    cars_df = pd.concat([cars_df, train_model_hashed_df], axis=1)

    # Drop the model feature
    cars_df.drop(columns='model', inplace=True)

    # Apply exterior_color transformation
    print("Apply exterior_color transformation")
    # Apply lower case and remove special characters
    cars_df['exterior_color'] = cars_df['exterior_color'].apply(clean_exterior_color)
    # Tokenize colors sentences
    tokenized_exterior_color = [simple_preprocess(sentence) for sentence in cars_df['exterior_color'].tolist()]
    # Train the Word2Vec model
    w2v_exterior_color_model = Word2Vec(sentences=tokenized_exterior_color, vector_size=exterior_color_vector_size,
                                        window=5, min_count=1, workers=4)
    # Calculate the vector for each interior color
    train_exterior_color_vectors_s = cars_df['exterior_color'].apply(
        lambda ic: get_exterior_color_phrase_vector(ic, w2v_exterior_color_model))
    # Replace the nan values with an array of (0,0,0)
    base_invalid_value = [0] * exterior_color_vector_size
    train_exterior_color_vectors_s = train_exterior_color_vectors_s.apply(
        lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    # Generate the interior color df using the transformed feature vectors
    train_exterior_color_df = pd.DataFrame(train_exterior_color_vectors_s.values.tolist(),
                                           columns=[f'exterior_color_x{i}' for i in
                                                    range(len(train_exterior_color_vectors_s.iloc[0]))],
                                           index=cars_df.index)
    # Concatenate the dataframes
    cars_df = pd.concat([cars_df, train_exterior_color_df], axis=1)

    # Once used drop the exterior_color feature
    cars_df.drop(columns='exterior_color', inplace=True)

    # Apply interior_color transformation
    print("Apply interior_color transformation")
    # Apply lower case and remove special characters
    cars_df['interior_color'] = cars_df['interior_color'].apply(clean_interior_color)
    # Tokenize colors sentences
    tokenized_interior_color = [simple_preprocess(sentence) for sentence in cars_df['interior_color'].tolist()]
    # Train the Word2Vec model
    w2v_interior_color_model = Word2Vec(sentences=tokenized_interior_color, vector_size=interior_color_vector_size,
                                        window=5, min_count=1, workers=4)
    # Calculate the vector for each interior color
    train_interior_color_vectors_s = cars_df['interior_color'].apply(
        lambda ic: get_interior_color_phrase_vector(ic, w2v_interior_color_model))
    # Replace the nan values with an array of (0,0,0)
    base_invalid_value = [0] * interior_color_vector_size
    train_interior_color_vectors_s = train_interior_color_vectors_s.apply(
        lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    # Generate the interior color df using the transformed feature vectors
    train_interior_color_df = pd.DataFrame(train_interior_color_vectors_s.values.tolist(),
                                           columns=[f'interior_color_x{i}' for i in
                                                    range(len(train_interior_color_vectors_s.iloc[0]))],
                                           index=cars_df.index)
    # Concatenate the dataframes
    cars_df = pd.concat([cars_df, train_interior_color_df], axis=1)
    # Once used drop the interior_color feature
    cars_df.drop(columns='interior_color', inplace=True)

    # Applt drive train transformation
    print("Apply drivetrain transformation")
    cars_df['drivetrain'] = cars_df['drivetrain'].map(map_drivetrain)
    # Initialize the OneHotEncoder drivetrain
    drivetrain_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit and transform the data
    ohe_drivetrain_model = drivetrain_encoder.fit(cars_df[['drivetrain']])
    train_drivetrain_encoded_data = ohe_drivetrain_model.transform(cars_df[['drivetrain']])
    # Convert the drivetrain encoded data into a DataFrame
    train_drivetrain_encoded_df = pd.DataFrame(train_drivetrain_encoded_data,
                                               columns=drivetrain_encoder.get_feature_names_out(['drivetrain']),
                                               index=cars_df.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    cars_df = pd.concat([cars_df, train_drivetrain_encoded_df], axis=1)

    # Once used drop the interior_color feature
    cars_df.drop(columns='drivetrain', inplace=True)

    # Apply make transformation
    print("Apply make transformation")
    # Initialize the OneHotEncoder maker
    make_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit and transform the data
    ohe_make_model = make_encoder.fit(cars_df[['make']])
    train_make_encoded_data = ohe_make_model.transform(cars_df[['make']])
    # Convert the drivetrain encoded data into a DataFrame
    train_make_encoded_df = pd.DataFrame(train_make_encoded_data, columns=make_encoder.get_feature_names_out(['make']),
                                         index=cars_df.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    cars_df = pd.concat([cars_df, train_make_encoded_df], axis=1)

    # Once used drop the make feature
    cars_df.drop(columns='make', inplace=True)

    # Apply bodystyle transformation
    print("Apply bodystyle transformation")
    # Initialize the OneHotEncoder bodystyle
    bodystyle_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit and transform the data
    ohe_bodystyle_model = bodystyle_encoder.fit(cars_df[['bodystyle']])
    train_bodystyle_encoded_data = ohe_bodystyle_model.transform(cars_df[['bodystyle']])
    # Convert the drivetrain encoded data into a DataFrame
    train_bodystyle_encoded_df = pd.DataFrame(train_bodystyle_encoded_data,
                                              columns=bodystyle_encoder.get_feature_names_out(['bodystyle']),
                                              index=cars_df.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    cars_df = pd.concat([cars_df, train_bodystyle_encoded_df], axis=1)

    # Once used drop the interior_color feature
    cars_df.drop(columns='bodystyle', inplace=True)

    # Apply cat transformation
    print("Apply cat transformation")
    # Apply lower case and remove special characters
    cars_df['cat'] = cars_df['cat'].apply(clean_cat)
    # Tokenize colors sentences
    tokenized_cat = [simple_preprocess(sentence) for sentence in cars_df['cat'].tolist()]
    # Train the Word2Vec model
    w2v_cat_model = Word2Vec(sentences=tokenized_cat, vector_size=cat_vector_size, window=5, min_count=1, workers=4)
    # Calculate the vertor for each cat
    train_cat_vectors_s = cars_df['cat'].apply(lambda ic: get_cat_phrase_vector(ic, w2v_cat_model))
    # Replace the nan values with an array of (0,0,0)
    base_invalid_value = [0] * cat_vector_size
    train_cat_vectors_s = train_cat_vectors_s.apply(lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    # Generate the interior color df using the transformed feature vectors
    train_cat_data = pd.DataFrame(train_cat_vectors_s.values.tolist(),
                                  columns=[f'cat_x{i}' for i in range(len(train_cat_vectors_s.iloc[0]))],
                                  index=cars_df.index)

    # Concatenate the dataframes
    cars_df = pd.concat([cars_df, train_cat_data], axis=1)

    # Remove cat column
    cars_df.drop(columns='cat', inplace=True)

    # Apply fuel type transformation
    print("Apply fuel_type transformation")
    cars_df['fuel_type'] = cars_df['fuel_type'].map(map_fuel_type)
    # Remove invalid fuel type
    cars_df = cars_df.loc[~cars_df['fuel_type'].isna()]

    # Initialize the OneHotEncoder drivetrain
    eho_fuel_type_model = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit and transform the data
    ohe_fuel_type_model = eho_fuel_type_model.fit(cars_df[['fuel_type']])
    train_fuel_type_encoded_data = ohe_fuel_type_model.transform(cars_df[['fuel_type']])
    # Convert the drivetrain encoded data into a DataFrame
    train_fuel_type_encoded_df = pd.DataFrame(train_fuel_type_encoded_data,
                                              columns=eho_fuel_type_model.get_feature_names_out(['fuel_type']),
                                              index=cars_df.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    cars_df = pd.concat([cars_df, train_fuel_type_encoded_df], axis=1)

    # Once used drop the interior_color feature
    cars_df.drop(columns='fuel_type', inplace=True)

    # Apply binary transformation
    print("Apply stock_type transformation")
    cars_df['stock_type'] = cars_df['stock_type'].map(map_stock_type)

    print(f"Split Dataset train-test_m. train={1 - test_size}, test_m={test_size}")
    train_cars_df, test_cars_df = train_test_split(cars_df, test_size=test_size, random_state=42)

    print("####### Imputate data")
    # train/use Impute
    print("Apply Iterative imputation")
    # NOTE: This condition is needed because the imputation model take a long training
    imputer_model_filename = 'preprocess_regression_imputer_model.pkl'
    if train_inputer:
        # Train imputer
        imp = IterativeImputer(estimator=RandomForestRegressor(), verbose=1)
        # fit on the dataset 
        imp.fit(train_cars_df)
        # Save imnputer model
        joblib.dump(imp, os.path.join(ARTIFACTS_FOLDER_PATH, imputer_model_filename))

    # Load your model
    imp: IterativeImputer = joblib.load(os.path.join(ARTIFACTS_FOLDER_PATH, imputer_model_filename))

    # Apply imputation
    train_df_trans = imp.transform(train_cars_df)
    test_df_trans = imp.transform(test_cars_df)
    # transform the dataset 
    train_cars_df = pd.DataFrame(train_df_trans, columns=train_cars_df.columns, index=train_cars_df.index)
    test_cars_df = pd.DataFrame(test_df_trans, columns=test_cars_df.columns, index=test_cars_df.index)

    print("####### Remove outliers data")
    # ### Outliers Removal
    print("Apply Outlier Removal")
    iso_forest = IsolationForest(n_estimators=200, contamination=isolation_forest_contamination, random_state=42,
                                 verbose=1)
    # Fit the model
    iso_forest.fit(train_cars_df)
    # Remove outliers 
    train_cars_df['outlier'] = iso_forest.predict(train_cars_df)
    test_cars_df['outlier'] = iso_forest.predict(test_cars_df)
    # Remove global outliers
    train_cars_df = train_cars_df[train_cars_df['outlier'] != -1]
    test_cars_df = test_cars_df[test_cars_df['outlier'] != -1]
    # Remove the outlier column
    train_cars_df.drop(columns='outlier', inplace=True)
    test_cars_df.drop(columns='outlier', inplace=True)

    print("####### Scale data")
    # Init scaler model
    scaler = MinMaxScaler()
    scaler.fit(train_cars_df)
    print("Apply Scale Min/Max Transformation")
    # Apply scale transformation
    train_df_trans = scaler.transform(train_cars_df)
    test_df_trans = scaler.transform(test_cars_df)
    # transform the dataset
    train_cars_df = pd.DataFrame(train_df_trans, columns=train_cars_df.columns, index=train_cars_df.index)
    test_cars_df = pd.DataFrame(test_df_trans, columns=test_cars_df.columns, index=test_cars_df.index)

    print(f"Cars train dataset size after preprocess: {train_cars_df.shape}")

    # Filter the original data
    cars_o_train_df = cars_o_df.loc[train_cars_df.index]
    cars_o_test_df = cars_o_df.loc[test_cars_df.index]

    # Save preprocess models
    # Save drive train encoding model
    print("Save preprocess models")

    # Save Hasher model
    hasher_model_model_filename = 'preprocess_hasher_model_model.pkl'
    joblib.dump(hasher_model_model, os.path.join(ARTIFACTS_FOLDER_PATH, hasher_model_model_filename))
    # Save Drive train encoding model
    ohe_drivetrain_model_filename = 'preprocess_ohe_drivetrain_model.pkl'
    joblib.dump(ohe_drivetrain_model, os.path.join(ARTIFACTS_FOLDER_PATH, ohe_drivetrain_model_filename))
    # Save make encoding model
    ohe_make_model_filename = 'preprocess_ohe_make_model.pkl'
    joblib.dump(ohe_make_model, os.path.join(ARTIFACTS_FOLDER_PATH, ohe_make_model_filename))
    # Save bodystyle encoding model
    ohe_bodystyle_model_filename = 'preprocess_ohe_bodystyle_model.pkl'
    joblib.dump(ohe_bodystyle_model, os.path.join(ARTIFACTS_FOLDER_PATH, ohe_bodystyle_model_filename))
    # Save bodystyle encoding model
    ohe_fuel_type_model_filename = 'preprocess_ohe_fuel_type_model.pkl'
    joblib.dump(ohe_fuel_type_model, os.path.join(ARTIFACTS_FOLDER_PATH, ohe_fuel_type_model_filename))
    # Save exterior color encoding model
    w2v_exterior_color_model_filename = 'preprocess_w2v_exterior_color_model.model'
    w2v_exterior_color_model.save(os.path.join(ARTIFACTS_FOLDER_PATH, w2v_exterior_color_model_filename))
    # Save interior color encoding model
    w2v_interior_color_model_filename = 'preprocess_w2v_interior_color_model.model'
    w2v_interior_color_model.save(os.path.join(ARTIFACTS_FOLDER_PATH, w2v_interior_color_model_filename))
    # Save exterior color encoding model
    w2v_cat_model_filename = 'preprocess_w2v_cat_model.model'
    w2v_cat_model.save(os.path.join(ARTIFACTS_FOLDER_PATH, w2v_cat_model_filename))
    # NOTE: imputer model was already saved
    # Save isolation forest model
    iso_forest_model_filename = 'preprocess_outlier_detection_model.pkl'
    joblib.dump(iso_forest, os.path.join(ARTIFACTS_FOLDER_PATH, iso_forest_model_filename))
    # Save scaler model
    scaler_model_filename = 'preprocess_scaler_model.pkl'
    joblib.dump(scaler, os.path.join(ARTIFACTS_FOLDER_PATH, scaler_model_filename))

    # Save data files
    cars_o_train_filepath = os.path.join(DATA_FOLDER_PATH, 'preprocess', 'cars_o_train.csv')
    cars_o_train_df.to_csv(cars_o_train_filepath, index=False)
    cars_train_preprocessed_filepath = os.path.join(DATA_FOLDER_PATH, 'preprocess', 'cars_train_preprocessed.csv')
    train_cars_df.to_csv(cars_train_preprocessed_filepath, index=False)

    cars_o_test_filepath = os.path.join(DATA_FOLDER_PATH, 'preprocess', 'cars_o_test.csv')
    cars_o_test_df.to_csv(cars_o_test_filepath, index=False)
    cars_test_preprocessed_filepath = os.path.join(DATA_FOLDER_PATH, 'preprocess', 'cars_test_preprocessed.csv')
    test_cars_df.to_csv(cars_test_preprocessed_filepath, index=False)

    # Build preprocess data report
    preprocess_config_data = {
        'data': {
            'cars_o_train_filepath': cars_o_train_filepath,
            'cars_train_preprocessed_filepath': cars_train_preprocessed_filepath,
            'cars_o_test_filepath': cars_o_test_filepath,
            'cars_test_preprocessed_filepath': cars_test_preprocessed_filepath,
        },
        'preprocess_config': {
            'price_threshold': price_threshold,
            'make_valid_categories': list(make_valid_categories),
            'exterior_color_vector_size': exterior_color_vector_size,
            'interior_color_vector_size': interior_color_vector_size,
            'cat_vector_size': cat_vector_size
        },
        'models_filenames': {
            'imputer_model_filename': imputer_model_filename,
            'scaler_model_filename': scaler_model_filename,
            'outlier_detection_filename': iso_forest_model_filename,
            'hasher_model_model_filename': hasher_model_model_filename,
            'ohe_drivetrain_model_filename': ohe_drivetrain_model_filename,
            'ohe_make_model_filename': ohe_make_model_filename,
            'ohe_bodystyle_model_filename': ohe_bodystyle_model_filename,
            'ohe_fuel_type_model_filename': ohe_fuel_type_model_filename,
            'w2v_exterior_color_model_filename': w2v_exterior_color_model_filename,
            'w2v_interior_color_model_filename': w2v_interior_color_model_filename,
            'w2v_cat_model_filename': w2v_cat_model_filename
        }
    }
    preprocess_config_filename = 'preprocess_config.json'
    with open(os.path.join(ARTIFACTS_FOLDER_PATH, preprocess_config_filename), 'w') as json_file:
        json.dump(preprocess_config_data, json_file)

    print("Preprocess completed")

    # Return model
    return preprocess_config_data

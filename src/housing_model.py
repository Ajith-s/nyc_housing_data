import sys

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor


# Load the dataset
def load_data():
    # Load your dataset here
    df = pd.read_csv('data/NY-House-Dataset.csv')
    return df


categorical_features = ['TYPE', 'ZIP_CODE', 'SUBLOCALITY']
continuous_features = ['BEDS', 'BATH', 'PROPERTYSQFT']


# Perform data transformations
def perform_data_transformations(df, model_columns):
    # Add your data transformations here
    df['ZIP_CODE'] = df['STATE'].str.extract(r'(\d{5})')
    df = df[df.loc[:, "PRICE"] < 100000000]
    for feature in categorical_features:
        df[feature] = df[feature].astype('category')
    model_df = df.loc[:, model_columns]
    return model_df


def encode_data(df, categorical_features, continuous_features):
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    X = df_encoded[
        continuous_features + df_encoded.columns[df_encoded.columns.str.startswith('TYPE_')].tolist() +
        df_encoded.columns[df_encoded.columns.str.startswith('ZIP_CODE_')].tolist() + df_encoded.columns[
            df_encoded.columns.str.startswith('SUBLOCALITY_')].tolist()].astype(float)
    return X


def scale_data(X):
    sqft_scaled = StandardScaler()
    X['PROPERTYSQFT'] = X.loc[:, 'PROPERTYSQFT'].sqft_scaled.fit_transform(X[['PROPERTYSQFT']])
    current_dir = os.path.dirname("/Users/ajiths/PycharmProjects/NYC_home_price_project/src")
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    joblib.dump(sqft_scaled, scaler_path)
    return X


def train_test_split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_random_forest_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [20, 30, 35],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    current_dir = os.path.dirname("/Users/ajiths/PycharmProjects/NYC_home_price_project/src")
    model_path = os.path.join(current_dir, 'rf_model.pkl')
    joblib.dump(model, model_path)
    return model


# Train the LightGBM Regressor
def train_lightgbm_regressor(X_train, y_train, model_path='light_gbm.pkl'):
    lgbm_model = LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=100, objective='regression')
    lgbm_model.fit(X_train, y_train)
    joblib.dump(lgbm_model, model_path)
    return lgbm_model


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse


# Function to make predictions
def predict_price(beds, bath, prop_sqft, prop_type, zip_code, sublocality, model_path):
    # Load the pre-trained model
    model = joblib.load(model_path)

    # Load the pre-trained scaler
    # joblib.load('src/scaler.pkl')

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'BEDS': [beds],
        'BATH': [bath],
        'PROPERTYSQFT': [prop_sqft],
        'TYPE': [prop_type],
        'ZIP_CODE': [zip_code],
        'SUBLOCALITY': [sublocality]
    })

    # Make predictions
    for feature in categorical_features:
        input_data[feature] = input_data[feature].astype('category')
    prediction = model.predict(input_data)
    return prediction[0]


# Function to load the pre-trained model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully. Model type: {type(model)}")
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None


def main():
    # Load data
    # Button to trigger prediction

    df = load_data()

    # Perform data transformations
    model_columns = ['BEDS', 'BATH', 'PROPERTYSQFT', 'TYPE', 'ZIP_CODE', 'SUBLOCALITY', 'PRICE']
    df = perform_data_transformations(df, model_columns)

    light_gbm_path = 'light_gbm.pkl'

    # Streamlit app
    st.title('Home Price Prediction App')
    #
    # Input fields for user to enter property details
    beds = st.slider('Number of Bedrooms', min_value=1, max_value=50, value=3)
    bath = st.slider('Number of Bathrooms', min_value=1, max_value=50, value=2)
    prop_sqft = st.slider('Property Square Footage', min_value=500, max_value=100000, value=2000)
    prop_type = st.selectbox('Property Type', df['TYPE'].unique())
    zip_code = st.text_input('Zip Code', 'Enter Zip Code')
    sublocality = st.text_input('Sublocality', 'Enter Sub Locality')

    if st.button('Predict Price'):

        # Make predictions using the pre-trained model

        prediction = predict_price(beds, bath, prop_sqft, prop_type, zip_code, sublocality, light_gbm_path)
        st.success(f'Predicted Price: ${prediction:,.2f}')


if __name__ == "__main__":
    main()

#######################

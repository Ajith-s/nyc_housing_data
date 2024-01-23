import joblib
import pandas as pd
import numpy as np
import streamlit as st
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Load the dataset
def load_data():
    # Load your dataset here
    df = pd.read_csv('data/NY-House-Dataset.csv')
    return df


# Perform data transformations
def perform_data_transformations(df, model_columns):
    # Add your data transformations here
    df['ZIP_CODE'] = df['STATE'].str.extract(r'(\d{5})')
    df = df[df.loc[:, "PRICE"] < 100000000]
    model_df = df.loc[:, model_columns]
    return model_df


def encode_data(df, categorical_features, continuous_features):
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    X = df_encoded[continuous_features + df_encoded.columns[df_encoded.columns.str.startswith('TYPE_')].tolist() +
                   df_encoded.columns[df_encoded.columns.str.startswith('ZIP_CODE_')].tolist() + df_encoded.columns[
                       df_encoded.columns.str.startswith('SUBLOCALITY_')].tolist()].astype(float)
    return X


def scale_data(X):
    sqft_scaled = StandardScaler()
    X['PROPERTYSQFT'] = sqft_scaled.fit_transform(X[['PROPERTYSQFT']])
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


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse


# Function to make predictions
def predict_price(beds, bath, prop_sqft, prop_type, zip_code, sublocality, model_path):
    # Load the pre-trained model
    rf_model = joblib.load(model_path)

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


    # Encode and scale the input data
    encoded_data = encode_data(input_data, categorical_features, continuous_features)
    scaled_data = scale_data(encoded_data)


    # Make predictions
    prediction = rf_model.predict(scaled_data)
    return prediction[0]


# rf_model = joblib.load('/Users/ajiths/PycharmProjects/NYC_home_price_project/src/rf_model.pkl')
# scaler = joblib.load('src/scaler.pkl')

# Main function to orchestrate the workflow

def main():
    # Load data
    # Button to trigger prediction

    df = load_data()

    # Perform data transformations
    model_columns = ['BEDS', 'BATH', 'PROPERTYSQFT', 'TYPE', 'ZIP_CODE', 'SUBLOCALITY', 'PRICE']
    df = perform_data_transformations(df, model_columns)
    # Define categorical features for one-hot encoding
    categorical_features = ['TYPE', 'ZIP_CODE', 'SUBLOCALITY']
    continuous_features = ['BEDS', 'BATH', 'PROPERTYSQFT']
    # One-hot encode categorical features
    encoded_data = encode_data(df, categorical_features, continuous_features)
    scaled_data = scale_data(encoded_data)

    y = df['PRICE']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_data,y)

    # Train the Random Forest model
    model = train_random_forest_model(X_train, y_train)

    # Evaluate the model
    rmse = evaluate_model(model, X_test, y_test)
    print(f'Root Mean Squared Error: {rmse}')


if __name__ == "__main__":
    main()


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


# ---- old code
# def main():
#     st.title('Home Price Prediction App')
#
#     # Input fields for user to enter property details
#     beds = st.slider('Number of Bedrooms', min_value=1, max_value=50, value=3)
#     bath = st.slider('Number of Bathrooms', min_value=1, max_value=50, value=2)
#     prop_sqft = st.slider('Property Square Footage', min_value=500, max_value=100000, value=2000)
#     prop_type = st.selectbox('Property Type', df['TYPE'].unique())
#     zip_code = st.text_input('Zip Code', 'Enter Zip Code')
#     sublocality = st.text_input('Sublocality', 'Enter Sub Locality')
#
#     if st.button('Predict Price'):
#         # Make predictions using the pre-trained model
#         current_dir = os.path.dirname(os.path.abspath('/Users/ajiths/PycharmProjects/NYC_home_price_project/src'))
#         model_path = os.path.join(current_dir, 'rf_model.pkl')
#         prediction = predict_price(beds, bath, prop_sqft, prop_type, zip_code, sublocality, model_path)
#         st.success(f'Predicted Price: ${prediction:,.2f}')
#
#
# if __name__ == "__main__":
#     # Load data
#     df = load_data()
#
#     # Perform data transformations
#     model_columns = ['BEDS', 'BATH', 'PROPERTYSQFT', 'TYPE', 'ZIP_CODE', 'SUBLOCALITY', 'PRICE']
#     df = perform_data_transformations(df, model_columns)
#
#     # Define categorical features for one-hot encoding
#     categorical_features = ['TYPE', 'ZIP_CODE', 'SUBLOCALITY']
#     continuous_features = ['BEDS', 'BATH', 'PROPERTYSQFT']
#
#     # One-hot encode categorical features
#     encoded_data = encode_data(df, categorical_features, continuous_features)
#     scaled_data = scale_data(encoded_data)
#
#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(scaled_data, df['PRICE'])
#
#     # Train the Random Forest model if it's not already trained
#     if not os.path.exists('rf_model.pkl'):
#         model = train_random_forest_model(X_train, y_train)
#         rmse = evaluate_model(model, X_test, y_test)
#         print(f'Root Mean Squared Error: {rmse}')
#
#     # Run the Streamlit app
#     main()
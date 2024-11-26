import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

import category_encoders as ce
from textblob import TextBlob

import pickle


version = "v2"
MODEL_FILE = f"movie_rating_pred_{version}.bin"


def read_prep_data(df):

    # Select Columns of Interest
    cols_of_interest = [
        "vote_average",
        "vote_count",
        "status",
        "release_date",
        "revenue",
        "runtime",
        "budget",
        "overview",
        "popularity",
        "genres",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "keywords",
    ]

    df = df[cols_of_interest]
    df = df[df.vote_average > 0]

    # Extract release year and age of the movie
    df["release_year"] = pd.to_datetime(df.release_date).dt.year
    df["movie_age"] = 2024 - df["release_year"]
    df["overview_word_count"] = df["overview"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    df["overview_sentiment"] = df["overview"].apply(
        lambda x: TextBlob(x).sentiment.polarity if pd.notna(x) else 0
    )

    # Replace missing values
    df.fillna(0, inplace=True)

    # Log transformation for skewed numerical features
    df["log_budget"] = np.log1p(df["budget"])
    df["log_revenue"] = np.log1p(df["revenue"])
    df["log_popularity"] = np.log1p(df["popularity"])

    # Drop columns we no longer need
    df.drop(
        columns=["release_date", "release_year", "budget", "revenue", "popularity"],
        inplace=True,
    )
    print(f"Data Prep complete!")

    return df


if __name__ == "__main__":

    print(f"Reading TMDB Movies Data.. ")
    raw_df = pd.read_csv("tmdb-movies.csv")

    df = read_prep_data(raw_df)

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_full_train = df_full_train.reset_index(drop=True)

    y_train = df_train["vote_average"].values
    y_val = df_val["vote_average"].values
    y_test = df_test["vote_average"].values
    y_full_train = df_full_train["vote_average"].values

    del df_train["vote_average"]
    del df_val["vote_average"]
    del df_test["vote_average"]
    del df_full_train["vote_average"]

    print("Train, Validation, Test - Data Shape")
    print(df_train.shape, df_val.shape, df_test.shape)

    # List of features for processing
    numerical_features = [
        "runtime",
        "vote_count",
        "log_budget",
        "log_revenue",
        "log_popularity",
        "movie_age",
        "overview_word_count",
        "overview_sentiment",
    ]
    categorical_features_small = ["status"]
    high_cardinality_features = [
        "genres",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "keywords",
    ]

    # Target Encoding for High Cardinality Features
    target_enc = ce.TargetEncoder(cols=high_cardinality_features)
    df_train_encoded = target_enc.fit_transform(df_train, y_train)
    df_test_encoded = target_enc.transform(df_test)

    df_train = df_train_encoded.reset_index(drop=True)
    df_test = df_test_encoded.reset_index(drop=True)

    # Create Pipelines for Numerical and Categorical Features

    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="mean"),
            ),  # Fill missing values with mean
            ("scaler", StandardScaler()),  # Standardize numerical values
        ]
    )

    # Categorical Transformer for One-Hot Encoding (for low cardinality categorical features)
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Column Transformer to apply transformations to different features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features_small),
        ],
        remainder="drop",  # Drop any columns not specified
    )

    # Final Model
    print(" Preprocessing and Modelling ..")

    X_full_train_pp = preprocessor.fit_transform(df_full_train)
    X_test_pp = preprocessor.transform(df_test)

    dfulltrain = xgb.DMatrix(X_full_train_pp, label=y_full_train)
    dtest = xgb.DMatrix(X_test_pp, label=y_test)

    xgb_params = {
        "eta": 0.1,
        "max_depth": 7,
        "min_child_weight": 10,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "nthread": 8,
        "seed": 1,
        "verbosity": 1,
        "lambda": 1,  # L2 regularization
        "alpha": 1,  # L1 regularization
    }

    evals = [(dfulltrain, "train"), (dtest, "test")]
    evals_result = {}

    model = xgb.train(
        xgb_params,
        dfulltrain,
        num_boost_round=150,
        early_stopping_rounds=10,
        evals=evals,
        evals_result=evals_result,
        verbose_eval=False,
    )

    test_pred = model.predict(dtest)
    rmse = root_mean_squared_error(test_pred, y_test)
    print(f"Model {version} training completed. RMSE: {rmse}")

    # Save model
    with open(MODEL_FILE, "wb") as f_out:
        pickle.dump((target_enc, preprocessor, model), f_out)

    print(f"Output Model saved to : {MODEL_FILE}")

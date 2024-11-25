import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import root_mean_squared_error


import xgboost as xgb
import pickle


version = "v1"
OUTPUT_FILE = f"movie_rating_pred_{version}.bin"
HASH_N_FEATURES = 50

xgb_params = {
    "eta": 0.1,
    "max_depth": 10,
    "min_child_weight": 100,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "nthread": 8,
    "seed": 1,
    "verbosity": 1,
    "lambda": 1,  # L2 regularization
    "alpha": 1,  # L1 regularization
}


def read_prep_data():
    print("Reading data.. ")
    df = pd.read_csv("tmdb-movies.csv")

    print("Preparing data..")

    cols_of_interest = [
        "title",
        "vote_average",
        "vote_count",
        "status",
        "release_date",
        "revenue",
        "runtime",
        "adult",
        "budget",
        "original_title",
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

    df["release_year"] = pd.to_datetime(df.release_date).dt.year
    mean_year = df.release_year.mean()
    df["release_year"] = df.release_year.fillna(mean_year)

    del df["release_date"]

    df["title"] = df.title.fillna("")
    df["original_title"] = df.original_title.fillna("")
    df["overview"] = df.overview.fillna("")
    df["genres"] = df.genres.fillna("")
    df["production_companies"] = df.production_companies.fillna("")
    df["production_countries"] = df.production_countries.fillna("")
    df["spoken_languages"] = df.spoken_languages.fillna("")
    df["keywords"] = df.keywords.fillna("")

    df = df.replace(0, np.nan)
    for col in ["vote_count", "runtime", "budget", "revenue", "popularity"]:
        col_mean = df[col].mean()
        df[col] = df[col].fillna(col_mean)

    return df


def train_test_data(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(f" Data Shape of Train:{df_full_train.shape}, Test: {df_test.shape}")

    y_full_train = df_full_train["vote_average"].values
    y_test = df_test["vote_average"].values

    del df_full_train["vote_average"]
    del df_test["vote_average"]

    # Pre-process data
    print("Preprocessing data..")

    hasher = FeatureHasher(n_features=HASH_N_FEATURES, input_type="dict")

    full_train_dicts = df_full_train.to_dict(orient="records")
    test_dicts = df_test.to_dict(orient="records")

    X_full_train = hasher.fit_transform(full_train_dicts)
    X_test = hasher.transform(test_dicts)

    X_full_train_dense = X_full_train.toarray()
    num_features = X_full_train_dense.shape[1]
    features = [f"f_{i}" for i in range(num_features)]

    dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)

    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
    # Build Model

    print(f"Building XGBoost Model with params:\n  {xgb_params}")

    model = xgb.train(xgb_params, dfulltrain, num_boost_round=200)

    y_pred = model.predict(dtest)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f" RMSE for the movies data model: {rmse}")

    # Save model
    with open(OUTPUT_FILE, "wb") as f_out:
        pickle.dump((hasher, model), f_out)

    print(f"Output Model saved to : {OUTPUT_FILE}")


if __name__ == "__main__":
    m_df = read_prep_data()
    train_test_data(m_df)

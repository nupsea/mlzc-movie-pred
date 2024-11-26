import pickle
import json
import xgboost as xgb
import numpy as np
import pandas as pd
from textblob import TextBlob
from flask import Flask, request, jsonify

# Flask app initialization
app = Flask("MovieRatingPredictorRefined")

version = "v2"
MODEL_FILE = f"movie_rating_pred_{version}.bin"

print(f"Loading Model from: {MODEL_FILE}")
with open(MODEL_FILE, "rb") as f_in:
    target_enc, preprocessor, model = pickle.load(f_in)


def predict_rating(movie):
    df = pd.DataFrame([movie])

    df_prepared = read_prep_data(df)

    X_enc = target_enc.transform(df_prepared)
    X = preprocessor.transform(X_enc)

    # Convert to DMatrix for XGBoost
    features = preprocessor.get_feature_names_out()
    features = features.tolist()
    d_eval = xgb.DMatrix(data=X, feature_names=features)

    y_pred = model.predict(d_eval)
    response = {"movie_title": movie["title"], "predicted_rating": float(y_pred[0])}

    return response


@app.route("/rate", methods=["POST"])
def rate():
    movie = request.get_json()

    if not movie:
        return jsonify({"error": "Invalid input"}), 400

    response = predict_rating(movie)
    print(f"Response : {response}")

    return jsonify(response)


def read_prep_data(df):
    # Select Columns of Interest
    cols_of_interest = [
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

    df = df[cols_of_interest].copy()

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
    # Direct call for testing
    movie = '{"id": 157336, "title": "Interstellar", "vote_count": 32571, "status": "Released", "release_date": "2014-11-05", "revenue": 701729206, "runtime": 169, "adult": false, "backdrop_path": "/pbrkL804c8yAv3zBZR4QPEafpAR.jpg", "budget": 165000000, "homepage": "http://www.interstellarmovie.net/", "imdb_id": "tt0816692", "original_language": "en", "original_title": "Interstellar", "overview": "The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.", "popularity": 140.241, "poster_path": "/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg", "tagline": "Mankind was born on Earth. It was never meant to die here.", "genres": "Adventure, Drama, Science Fiction", "production_companies": "Legendary Pictures, Syncopy, Lynda Obst Productions", "production_countries": "United Kingdom, United States of America", "spoken_languages": "English", "keywords": "rescue, future, spacecraft, race against time, artificial intelligence (a.i.), nasa, time warp, dystopia, expedition, space travel, wormhole, famine, black hole, quantum mechanics, family relationships, space, robot, astronaut, scientist, single father, farmer, space station, curious, space adventure, time paradox, thoughtful, time-manipulation, father daughter relationship, 2060s, cornfield, time manipulation, complicated"}'
    json_data = json.loads(movie)
    result = predict_rating(json_data)
    print(f"Result: {result}")

    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=5000)

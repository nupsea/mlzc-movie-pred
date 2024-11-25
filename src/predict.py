import pickle
import json
import xgboost as xgb

from flask import Flask, request, jsonify


app = Flask("MovieRatingPredictor")

version = "v1"
INPUT_FILE = f"movie_rating_pred_{version}.bin"


@app.route("/rate", methods=["POST"])
def rate():
    print(f"Reading Model from : {INPUT_FILE}")
    with open(INPUT_FILE, "rb") as f_in:
        hasher, model = pickle.load(f_in)

    movie = request.get_json()
    print(f"Input Movie metadata: {movie}")

    X = hasher.transform([movie]).toarray()
    num_features = X.shape[1]
    features = [f"f_{i}" for i in range(num_features)]
    d_eval = xgb.DMatrix(data=X, feature_names=features)
    y_pred = model.predict(d_eval)

    response = {"movie_title": movie["title"], "predicted_rating": float(y_pred)}

    return jsonify(response)


if __name__ == "__main__":
    movie = '{"title": "Cold River", "vote_count": 1.0, "status": "Released", "revenue": 43403720.66610969, "runtime": 94.0, "adult": false, "budget": 10065207.438372264, "original_title": "Cold River", "overview": "Based on the novel Winterkill, by William Judson, Cold River is the story of an Adirondack guide who takes his young daughter and step-son on a long camping trip in the fall of 1932. When winter strikes unexpectedly early (a natural phenomenon known as a \'winterkill\' - so named because the animals are totally unprepared for a sudden, early winter, and many freeze or starve to death), a disastrous turn of events leaves the two children to find their own way home without food, or protection from the elements.", "popularity": 1.5, "genres": "Adventure", "production_companies": "", "production_countries": "United States of America", "spoken_languages": "English", "keywords": "winter, camping", "release_year": 1982.0}'
    json_data = json.loads(movie)
    result = rate(json_data)
    print(f"Result: {result}")

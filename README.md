# MLZoomcamp Movie Rating Predictor

## Overview

This project aims to train a Machine Learning (ML) model that predicts the rating of a movie based on various features such as movie title, release year, runtime, revenue, popularity, overview, and other relevant metadata. The model serves as a useful tool for recommender systems, film analysis, or providing quick movie rating estimates for new releases.

The model is built using XGBoost, a powerful gradient boosting library, and is served as a REST API using the Flask framework. The API can be used to get predicted ratings for movies given appropriate input features.

## Dataset

The dataset used for this project is sourced from Kaggle:

- **Source**: [TMDB Movies Dataset 2023 - 930k Movies](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
- **Description**: This dataset contains metadata of approximately 930,000 movies, including features like movie titles, production companies, genres, release dates, budgets, and more.

## Setup / Installation

### Clone the Repository
First, clone this repository to your local machine and navigate to it:
```bash
# Clone the repository
git clone <repository-url>
cd movie-rating-predictor
```

### Install Dependencies
To set up the environment, you will need to install the necessary dependencies. It is recommended to use `poetry` or any virtual environment tool of your choice.

#### Using Poetry
- If `poetry` is not installed, install it from [Poetry Installation Guide](https://python-poetry.org/docs/).
- Use the following command to install dependencies:

```bash
poetry install
```

This command will create an isolated environment and install all required packages, enabling you to run the notebooks and the Flask API server.

## Running with Docker

For ease of deployment, a Docker container is provided to package the environment and the model:

### Build Docker Image
```bash
# Build Docker Image
docker build -t movie-rating-predictor .
```

### Run Docker Container
```bash
# Run Docker Container
docker run -it --rm -p 9696:9696 movie-rating-predictor
```

This will start the API server inside the container, listening on port `9696`.

## Usage

### Accessing the Prediction API in Python/Notebook Environment
You can interact with the API via HTTP requests. Below is a sample Python code to send a movie's metadata for rating prediction:

```python
import requests
import numpy as np

# Sample movie for prediction
movie = raw_df.iloc[23].to_dict()

# Replace NaN values with None to avoid JSON serialization issues
for key, value in movie.items():
    if isinstance(value, float) and np.isnan(value):
        movie[key] = None

# Define the API endpoint
host = "0.0.0.0:9696"
url = f'http://{host}/rate'

# Send POST request for prediction
response = requests.post(url, json=movie)

# Output the response
print(response.json())
```

### Sample Response
```json
{
  "movie_title": "I Am Omega",
  "predicted_rating": 5.78
}
```

### Example Input
The following is a sample movie input that can be used to test the API:

```python
movie = {
 'id': 157336,
 'title': 'Interstellar',
 'vote_average': 8.417,
 'vote_count': 32571,
 'status': 'Released',
 'release_date': '2014-11-05',
 'revenue': 701729206,
 'runtime': 169,
 'adult': False,
 'backdrop_path': '/pbrkL804c8yAv3zBZR4QPEafpAR.jpg',
 'budget': 165000000,
 'homepage': 'http://www.interstellarmovie.net/',
 'imdb_id': 'tt0816692',
 'original_language': 'en',
 'original_title': 'Interstellar',
 'overview': 'The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.',
 'popularity': 140.241,
 'poster_path': '/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg',
 'tagline': 'Mankind was born on Earth. It was never meant to die here.',
 'genres': 'Adventure, Drama, Science Fiction',
 'production_companies': 'Legendary Pictures, Syncopy, Lynda Obst Productions',
 'production_countries': 'United Kingdom, United States of America',
 'spoken_languages': 'English',
 'keywords': 'rescue, future, spacecraft, race against time, artificial intelligence (a.i.), nasa, time warp, dystopia, expedition, space travel, wormhole, famine, black hole, quantum mechanics, family relationships, space, robot, astronaut, scientist, single father, farmer, space station, curious, space adventure, time paradox, thoughtful, time-manipulation, father daughter relationship, 2060s, cornfield, time manipulation, complicated'
}
```

## Training the Model

If you need to train the model, you can modify the training script and run it locally:

### Train the Model
Make sure you are inside the poetry environment or have all dependencies installed.

```bash
poetry shell
python src/train2.py
```

This will retrain the model and save the updated version, which can then be used to rebuild the Docker image.

### Changing Model Code
To change the model code or update features, modify the following files:
- **Training Script**: `train[version].py`
- **Prediction Script**: `predict[version].py`

After modifying the code, rebuild and redeploy the Docker image to apply changes.

## Contributing
If you'd like to contribute to this project, feel free to open issues or submit pull requests. Contributions in improving the prediction accuracy, adding features, or enhancing documentation are always welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
- **XGBoost**: A powerful library for gradient boosting, used to build the model.
- **Flask**: Framework used for serving the model as a REST API.
- **TMDB Dataset**: Thanks to Kaggle for providing the movie dataset.

## Contact
For any questions or suggestions, please feel free to reach out to the repository maintainer through GitHub.

---


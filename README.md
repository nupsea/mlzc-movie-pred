# MLZoomcamp Movie Prediction

## About 

This project aims to train an ML model that predicts the rating of a movie based on various features such as the movie title, release year, runtime, revenue, popularity, overview, and other relevant metadata. The model can help provide an estimated rating for movies, useful for recommender systems or film analysis.
The ML model is built using XGBoost, a popular gradient boosting library, and the Flask framework is used to serve the model as a REST API endpoint.



## Setup / Installation 
Source Data: The dataset used for this project is sourced from Kaggle: TMDB Movies Dataset 2023 - 930k Movies. Download data from:
https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies 

To set up the project environment, you will need to install dependencies. It is recommended to use poetry or any virtual environment tool of your choice.
Clone the repository and navigate into it.
Install poetry (if not already installed).

Install the necessary Python packages using poetry.
```bash
poetry install 
```
This should enable you to go run through the notebook. To directly use the API, go with the below steps.


## Exec with Docker

```bash
# Build Docker Image
docker build -t rating-test .

# Run Docker Container
docker run -it --rm -p 9696:9696 rating-test
```

## Test in Notebook/Python Env
```python
import requests

# Select a movie from the dataset for prediction
movie = df_full_train.iloc[33].to_dict()

# Define host and endpoint
host = "0.0.0.0:9696"

# Send POST request for prediction
response = requests.post(
    url=f'http://{host}/rate',
    json=movie
)

# Output the response
print(response.json())

```

```python
# Sample data
{
     'title': 'Cold River',
     'vote_count': 1.0,
     'status': 'Released',
     'revenue': 43403720.66610969,
     'runtime': 94.0,
     'adult': False,
     'budget': 10065207.438372264,
     'original_title': 'Cold River',
     'overview': "Based on the novel Winterkill, by William Judson, Cold River is the story of an Adirondack guide who takes his young daughter and step-son on a long camping trip in the fall of 1932. When winter strikes unexpectedly early (a natural phenomenon known as a 'winterkill' - so named because the animals are totally unprepared for a sudden, early winter, and many freeze or starve to death), a disastrous turn of events leaves the two children to find their own way home without food, or protection from the elements.",
     'popularity': 1.5,
     'genres': 'Adventure',
     'production_companies': '',
     'production_countries': 'United States of America',
     'spoken_languages': 'English',
     'keywords': 'winter, camping',
     'release_year': 1982.0
}
```

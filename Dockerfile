FROM python:3.13-slim

# Install system dependencies required by Poetry and for building Python packages
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Update PATH to include poetry
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY ["pyproject.toml", "poetry.lock", "./"]

# Install dependencies using poetry
RUN poetry install --no-dev --no-interaction --no-root

COPY ["src/predict2.py", "movie_rating_pred_v2.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["poetry", "run", "gunicorn", "--bind=0.0.0.0:9696", "predict2:app"]



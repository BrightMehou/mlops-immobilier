FROM python:3.12.8-slim

# Installer les dépendances système nécessaires (dont curl)
RUN apt-get update && apt-get install -y curl

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="${PATH}:/root/.local/bin"

WORKDIR /app

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
COPY pyproject.toml poetry.lock /app/

RUN poetry install --no-root

# Copy the source code into the container.
COPY src /app/src
COPY app.py /app/app.py


# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD poetry run python src/train.py && poetry run uvicorn 'app:app' --host=0.0.0.0 --port=8000

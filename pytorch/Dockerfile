FROM nvcr.io/nvidia/pytorch:24.06-py3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# disable virtualenv creation
RUN poetry config virtualenvs.create false

COPY ../pyproject.toml ../poetry.lock ./
RUN poetry install --no-interaction --no-ansi --only main

COPY . .

CMD ["python", "main.py"]

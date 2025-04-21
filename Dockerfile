FROM python:3.10-slim

LABEL maintainer="choksijeet <jech4418@colorado.edu>"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p TASK1/outputs TASK2/outputs TASK3/outputs TASK4/outputs \
             TASK3/checkpoints TASK4/checkpoints

CMD ["bash"]

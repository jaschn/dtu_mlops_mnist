# Base image
FROM python:3.7-slim

   # install python
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/predict_model.py", "models", "data/processed/mnist", "reports/figures"]

#command: docker run --name predict -v $(pwd)/reports:/reports -v $(pwd)/models:/models predict:latest

# Base image
FROM python:3.7-slim

   # install python
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc wget && \
apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY .git/ .git/
COPY .dvc/config .dvc/config
COPY .dvc/plots .dvc/plots
COPY config/ config/
COPY data.dvc data.dvc
COPY docker_trainer_start.sh docker_trainer_start.sh

RUN mkdir data/
RUN mkdir models/
RUN mkdir reports/

RUN pip install dvc 'dvc[gs]'

RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["./docker_trainer_start.sh"]

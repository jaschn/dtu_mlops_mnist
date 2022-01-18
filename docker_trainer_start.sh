#!/bin/bash
dvc pull
python src/models/train_model.py $@
gsutil cp models/trained_model.pt gs://mlops_models_saves

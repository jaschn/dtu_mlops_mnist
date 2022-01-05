# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from dotenv import find_dotenv, load_dotenv

from src.data.dataset import mnist
from src.models.model import MyAwesomeModel


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("prediction_path", type=click.Path(exists=True))
@click.option("--cnt", default=10, type=int)
def main(model_filepath, image_path, prediction_path, cnt):
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    model_states = torch.load(os.path.join(model_filepath, "trained_model.pt"))
    model = MyAwesomeModel()
    model.load_state_dict(model_states)
    train_set = mnist(image_path, "test")
    images, labels = next(iter(train_set))
    out = model(images)
    ps = out.exp()
    _, top_class = ps.topk(1, dim=1)
    os.makedirs(prediction_path, exist_ok=True)
    for i, image in enumerate(images[:10]):
        plt.imshow(image[0], cmap="gray")
        plt.title(f"Prediction: {top_class[i].item()}, Label:{labels[i]}")
        plt.tick_params(axis="both", length=0)
        plt.savefig(os.path.join(prediction_path, f"image_{i}.png"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

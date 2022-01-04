# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


def process_train(input, output):
    """
    Stores the training data as pytorch file in the dir specified in output
            Parameters:
                    input (str): input directory of the corrupted data
                    output (str): where the processed data should be stored
    """
    base_path = os.path.join(input, "corruptmnist")
    with np.load(os.path.join(base_path, "train_0.npz")) as f:
        im = torch.from_numpy(f["images"]).float()
        # convert from (count, 28x28) to (count, 1 channel, 28x28)
        images = im.view(im.size(0), 1, im.size(1), im.size(2))
        labels = torch.from_numpy(f["labels"]).long()
    for i in range(1, 8):
        with np.load(os.path.join(base_path, f"train_{i}.npz")) as f:
            im = torch.from_numpy(f["images"]).float()
            im = im.view(im.size(0), 1, im.size(1), im.size(2))
            images = torch.cat([images, im], dim=0)
            lab = torch.from_numpy(f["labels"]).long()
            labels = torch.cat([labels, lab], dim=0)

    assert len(images) == len(labels), "image count does not match label count"

    output_file = os.path.join(output, "mnist", "train")
    os.makedirs(output_file, exist_ok=True)
    torch.save(images, os.path.join(output_file, "images.pt"))
    torch.save(labels, os.path.join(output_file, "labels.pt"))


def process_test(input, output):
    base_path = os.path.join(input, "corruptmnist")
    with np.load(os.path.join(base_path, "test.npz")) as f:
        im = torch.from_numpy(f["images"]).float()
        # convert from (count, 28x28) to (count, 1 channel, 28x28)
        images = im.view(im.size(0), 1, im.size(1), im.size(2))
        labels = torch.from_numpy(f["labels"]).long()

    assert len(images) == len(labels), "image count does not match label count"

    output_file = os.path.join(output, "mnist", "test")
    os.makedirs(output_file, exist_ok=True)
    torch.save(images, os.path.join(output_file, "images.pt"))
    torch.save(labels, os.path.join(output_file, "labels.pt"))


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    process_train(input_filepath, output_filepath)
    process_test(input_filepath, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

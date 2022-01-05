# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import find_dotenv, load_dotenv

from src.data.dataset import mnist
from src.models.model import MyAwesomeModel


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath_figure", type=click.Path())
@click.argument("output_filepath_model", type=click.Path())
@click.option("--lr", default=0.001, type=float)
@click.option("--epochs", default=10, type=int)
def main(input_filepath, output_filepath_figure, output_filepath_model, lr, epochs):
    """
    Trains the model

            Parameters:
                    input_filepath (str): filepath where the training data is loaded from
                    output_filepath_figure (str): where the graph of the loss is stored as a figure
                    output_filepath_model (str): where the weights of the model are stored
                    lr (float): learning rate for the optimizer
                    epochs (int): the number of epochs the training is run

            Returns:
                    binary_sum (str): Binary string of the sum of a and b
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    model = MyAwesomeModel()
    critirion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_set = mnist(input_filepath, "train")
    losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            out = model(images)
            loss = critirion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        losses.append(running_loss)
    plt.plot(losses, label="Loss")
    plt.legend()
    plt.title("Loss")
    os.makedirs(output_filepath_figure, exist_ok=True)
    plt.savefig(os.path.join(output_filepath_figure, "loss.png"))

    os.makedirs(output_filepath_model, exist_ok=True)
    torch.save(
        model.state_dict(), os.path.join(output_filepath_model, "trained_model.pt")
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

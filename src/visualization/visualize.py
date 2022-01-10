# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import torch
from dotenv import find_dotenv, load_dotenv
from sklearn.manifold import TSNE  # type: ignore

from src.data.dataset import mnist
from src.models.model import MyAwesomeModel


@hydra.main(config_path="./../../config", config_name="visualize_conf.yaml")
def main(cfg):
    model_filepath = os.path.join(hydra.utils.get_original_cwd(), cfg.model_filepath)
    image_path = os.path.join(hydra.utils.get_original_cwd(), cfg.image_path)
    visualize_path = cfg.visualize_path

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    model_states = torch.load(os.path.join(model_filepath, "trained_model.pt"))
    model = MyAwesomeModel(cfg.model)
    model.load_state_dict(model_states)

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.cnn[-1].register_forward_hook(get_activation("last_cnn"))

    train_set = mnist(image_path, "test")
    activation_list = []
    top_class_list = []
    for images, _ in train_set:
        out = model(images)
        activation_list.append(activation["last_cnn"])
        ps = out.exp()
        _, top_class = ps.topk(1, dim=1)
        top_class_list.append(top_class)

    activations = torch.cat(activation_list, dim=0)
    activations = activations.view(activations.size(0), -1)
    top_classes = torch.cat(top_class_list, dim=0)
    top_classes = top_classes.view(top_classes.size(0))
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(activations)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        palette=sns.color_palette("hls", 10),
        hue=top_classes,
        legend="full",
        alpha=0.3,
    )
    os.makedirs(visualize_path, exist_ok=True)
    plt.savefig(os.path.join(visualize_path, "visual.png"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

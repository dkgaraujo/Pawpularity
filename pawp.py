import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow_hub as tfhub

from utils import build_interm_models, sort_labels
from hub_models import model_handle_map, model_image_size_map

# Configurations

base_path = "data/"
imgs_path = "data/train"
models_path = "models"

hub_config = {
    "model_name": "efficientnetv2-s",  # ['efficientnetv2-s', 'efficientnetv2-m', 'efficientnetv2-l', 'efficientnetv2-s-21k', 'efficientnetv2-m-21k', 'efficientnetv2-l-21k', 'efficientnetv2-xl-21k', 'efficientnetv2-b0-21k', 'efficientnetv2-b1-21k', 'efficientnetv2-b2-21k', 'efficientnetv2-b3-21k', 'efficientnetv2-s-21k-ft1k', 'efficientnetv2-m-21k-ft1k', 'efficientnetv2-l-21k-ft1k', 'efficientnetv2-xl-21k-ft1k', 'efficientnetv2-b0-21k-ft1k', 'efficientnetv2-b1-21k-ft1k', 'efficientnetv2-b2-21k-ft1k', 'efficientnetv2-b3-21k-ft1k', 'efficientnetv2-b0', 'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'bit_s-r50x1', 'inception_v3', 'inception_resnet_v2', 'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'nasnet_large', 'nasnet_mobile', 'pnasnet_large', 'mobilenet_v2_100_224', 'mobilenet_v2_130_224', 'mobilenet_v2_140_224', 'mobilenet_v3_small_100_224', 'mobilenet_v3_small_075_224', 'mobilenet_v3_large_100_224', 'mobilenet_v3_large_075_224']
    "do_fine_tuning": False,
    "epochs": 3,
}

use_cached_models = True

# Preparing the dataset

train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
test_df = pd.read_csv(os.path.join(base_path, "test.csv"))
sample_sub = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))


train_df = sort_labels(
    train_df,
    "Id",
    os.path.join(base_path, "train"),
).sort_values(by="Id_for_ordering")

intermediary_outputs = train_df.columns[1:-2]
print(
    f"Those will be our intermediary output variables, which the models will be tweaked to recognise: {intermediary_outputs}"
)

# Transfer learning

model_handle = model_handle_map.get(hub_config["model_name"])
pixels = model_image_size_map.get(hub_config["model_name"], 224)

print(f"Selected model: {hub_config['model_name']} : {model_handle}")

IMAGE_SIZE = (pixels, pixels)
print(f"Input size {IMAGE_SIZE}")

BATCH_SIZE = 32

intermediate_models = []
for var in intermediary_outputs:
    model_name = f"{models_path}/{var}.tfmodel"
    if os.path.exists(model_name) & use_cached_models:
        print(f"{model_name} already exists. Loading it from file.")
        intermediate_models.append(tf.keras.models.load_model(model_name))
    else:
        model = build_interm_models(
            var,
            "data/train",
            train_df,
            model_handle,
            img_size=IMAGE_SIZE,
            hub_config=hub_config,
        )
        intermediate_models.append(model)
        model.save(model_name)

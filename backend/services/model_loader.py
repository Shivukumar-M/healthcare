import json
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer

def load_model(model_dir):
    model = TFSMLayer(
        f"{model_dir}/saved_model",
        call_endpoint="serve"
    )

    with open(f"{model_dir}/labels.json") as f:
        labels = json.load(f)

    return model, labels

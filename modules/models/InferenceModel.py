import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from modules.utils.load_model import load_model


class InferenceModelKNN():
    def __init__(self, emb_model_name, weights_path, knn_num=100, distance='cosine'):
        self.model = load_model(emb_model_name, weights_path)
        self.model.eval()

        self.knn = NearestNeighbors()
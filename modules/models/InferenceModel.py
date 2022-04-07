import torch
import numpy as np
import joblib
import cv2

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from tqdm import tqdm

from modules.utils.load_model import load_model


#def cosine_distance(X1, X2):
#    return 1.0 - cosine_similarity(X1.reshape(1, -1), X2.reshape(1, -1)).flatten()[0]


class InferenceModelKNN_Normalizer():
    def __init__(self,
                 model,
                 weights_path,
                 ohe_path,
                 img_preprocessor,
                 device,
                 knn_num=100,
                 distance='cosine'):
        self.device = device

        self.model = load_model(model, weights_path)
        self.model.to(device)
        self.model.eval()

        self.img_preprocessor = img_preprocessor
        self.ohe = joblib.load(ohe_path)
        self.knn = NearestNeighbors(n_neighbors=knn_num, metric=distance)
        self.targets = []


    def get_embeddings(self, dataset):
        embeds = []

        for data in tqdm(dataset, desc='Compute embeddings'):
            imgs_tensor = data['input'].to(self.device)
            with torch.no_grad():
                embeds_batch = self.model.forward_eval(imgs_tensor)
            embeds.append(embeds_batch.cpu().detach().numpy())

        a = np.concatenate(embeds, axis=0)
        return a


    def train(self, embeds, targets):
        # normalize
        #embeds += 1
        #embeds = normalize(embeds, axis=1, norm="l2")

        self.knn.fit(embeds)
        self.targets = np.array(targets)
        self.allow_targets = np.array(list(set(targets)))


    def predict(self, embeds, thr=0.2, print_quantile=False, mode='mean', k=5):
        # normalize
        #embeds += 1
        #embeds = normalize(embeds, axis=1, norm="l2")

        dists, indxs = self.knn.kneighbors(embeds, return_distance=True)
        #dists = 1 - dists
        preds_targets = np.array([self.targets[i] for i in indxs])

        if print_quantile:
            q = [x / 10 for x in range(1, 10, 1)]
            quantiles = [np.quantile(dists, q_) for q_ in q]
            print(f"Quantiles: {q}, Values: {quantiles}")

        groups = []
        for d, p in zip(dists, preds_targets):
            mapping_d, mapping_p = [], []
            for u in np.unique(p):
                u_indx = np.where(p == u)
                d_indx = d[u_indx]

                if mode == 'mean':
                    d_f = np.mean(d_indx)
                elif mode == 'min':
                    d_f = np.min(d_indx)

                mapping_d.append(d_f)
                mapping_p.append(u)

            mapping_d, mapping_p = np.array(mapping_d), np.array(mapping_p)
            mappings_arg_sorted = np.argsort(mapping_d)
            mapping_d, mapping_p = mapping_d[mappings_arg_sorted], mapping_p[mappings_arg_sorted]
            groups.append((mapping_d, mapping_p))

        # form predicts
        predicts = []
        for mapping_d, mapping_p in groups:
            new_ind_counter = 0
            pr = []
            for md, mp in zip(mapping_d[:k], mapping_p[:k]):
                if new_ind_counter == 0 and md >= thr:
                    pr.append('new_individual')
                    new_ind_counter += 1
                else:
                    pr.append(mp)

            predicts.append(pr)

        return predicts
import torch
import torch.nn as nn


from modules.models.ArcFaceHead import ArcMarginProduct
from modules.models.Effnet import EffNetBx, EffNetBx_v2


class EffB0_Arc(nn.Module):
    def __init__(self, n_classes=15587, s=30.0, m=0.30, easy_margin=False, eff_name='tf_efficientnet_b0'):
        super(EffB0_Arc, self).__init__()

        self.effnet = EffNetBx(model_name=eff_name)
        self.head = ArcMarginProduct(in_features=512, out_features=n_classes, s=s, m=m, easy_margin=easy_margin)

    def forward(self, input, label):
        embedding = self.effnet(input)
        cos_dist = self.head(embedding, label)
        return embedding, cos_dist

    def forward_eval(self, input):
        embedding = self.effnet(input)
        return embedding


# optimized for ArcFace
class EffB0_Arc_v2(nn.Module):
    def __init__(self, n_classes=15587, s=30.0, m=0.30, easy_margin=False, eff_name='tf_efficientnet_b0'):
        super(EffB0_Arc_v2, self).__init__()

        self.effnet = EffNetBx_v2(model_name=eff_name)
        self.head = ArcMarginProduct(in_features=512, out_features=n_classes, s=s, m=m, easy_margin=easy_margin)

    def forward(self, input, label):
        embedding = self.effnet(input)
        cos_dist = self.head(embedding, label)
        return embedding, cos_dist

    def forward_eval(self, input):
        embedding = self.effnet(input)
        return embedding
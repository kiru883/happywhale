import torch
import torch.nn as nn
import timm

from efficientnet_pytorch import EfficientNet
from modules.models.GEM_pooling import GeM


class EffNetBx(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0'):
        super(EffNetBx, self).__init__()

        if 'tf_efficientnet_b' in model_name:
            self.model = timm.create_model(model_name=model_name, pretrained=True, drop_rate=0.2)
            in_features = self.model.get_classifier().in_features

            self.model.classifier = nn.Identity()
            self.embed = nn.Linear(in_features=in_features, out_features=512)
            self.bn = nn.BatchNorm1d(num_features=512)

    def forward(self, inp):
        x = self.model(inp)
        x = self.embed(x)
        x = self.bn(x)
        return x


# save effnet but optimized for ArcFace
class EffNetBx_v2(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0'):
        super(EffNetBx_v2, self).__init__()

        if 'tf_efficientnet_b' in model_name:
            self.model = timm.create_model(model_name=model_name, pretrained=True, drop_rate=0.2)
            in_features = self.model.get_classifier().in_features

            self.model.classifier = nn.Identity()
            self.bn1 = nn.BatchNorm1d(num_features=in_features)
            self.drop = nn.Dropout(p=0.1)
            self.embed = nn.Linear(in_features=in_features, out_features=512)
            self.bn2 = nn.BatchNorm1d(num_features=512)

    def forward(self, inp):
        x = self.model(inp)
        x = self.bn1(x)
        x = self.drop(x)
        x = self.embed(x)
        x = self.bn2(x)
        return x



# save effnet but optimized for ArcFace
class EffNetBx_v3(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0'):
        super(EffNetBx_v3, self).__init__()

        if 'tf_efficientnet_b' in model_name:
            self.model = timm.create_model(model_name=model_name, pretrained=True, drop_rate=0.2)
            in_features = self.model.get_classifier().in_features

            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()

            self.gem_pooling = GeM()

            self.bn1 = nn.BatchNorm1d(num_features=in_features)
            self.drop = nn.Dropout(p=0.1)
            self.embed = nn.Linear(in_features=in_features, out_features=512)
            self.bn2 = nn.BatchNorm1d(num_features=512)

    def forward(self, inp):
        x = self.model(inp)
        # GEM POOLING
        x = self.gem_pooling(x).flatten(1)
        x = self.bn1(x)
        x = self.drop(x)
        x = self.embed(x)
        x = self.bn2(x)
        return x


# test
if __name__ == "__main__":
    test = EffNetBx()
    out = test(torch.rand(2, 3, 224, 224))
    print(test)
    print(out.shape)



import torch
import torch.nn as nn
import timm

from efficientnet_pytorch import EfficientNet


class EffNetBx(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0'):
        super(EffNetBx, self).__init__()

        if model_name == 'tf_efficientnet_b0':
            self.model = timm.create_model(model_name=model_name, pretrained=True, drop_rate=0.2)
            self.model.classifier = nn.Identity()
            self.embed = nn.Linear(in_features=1280, out_features=512)
            self.bn = nn.BatchNorm1d(num_features=512)

    def forward(self, inp):
        x = self.model(inp)
        x = self.embed(x)
        x = self.bn(x)
        return x


# test
if __name__ == "__main__":
    test = EffNetBx()
    out = test(torch.rand(2, 3, 224, 224))
    print(test)
    print(out.shape)



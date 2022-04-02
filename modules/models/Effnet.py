import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class EffNetBx(nn.Module):
    CLS_LAYERS = ['_fc', '_swish']

    def __init__(self, model_name='efficientnet-b0'):
        super(EffNetBx, self).__init__()

        if model_name == 'efficientnet-b0':
            self.model = EfficientNet.from_pretrained(model_name)
            # replace fc and swish layers, add NORM
            self.model._fc = nn.Identity()
            self.model._swish = nn.Identity()
            self.model = nn.Sequential(
                self.model,
                nn.Linear(in_features=1280, out_features=512),
                nn.BatchNorm1d(num_features=512)
            )

    def forward(self, inp):
        return self.model(inp)


# test
if __name__ == "__main__":
    test = EffNetBx()
    out = test(torch.rand(2, 3, 224, 224))
    print(out.shape)


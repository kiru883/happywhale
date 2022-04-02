import torch


def MaP5(predict, target):
    le = torch.argmax(target, dim=1)
    topk = torch.argsort(predict, dim=1, descending=True)[:, :5]

    match = (topk == le[..., None])
    match = match[match.any(axis=1)]
    match = torch.argmax(match.type(torch.int), dim=1) + 1

    return torch.sum(1 / match) / le.shape[0]


if __name__ == "__main__":
    a = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
    b = torch.tensor([[0.9, 0.05, 0.05], [0.01, 0.89, 0.1], [0.1, 0.01, 0.89], [0.1, 0.2, 0.6]])
    print(MaP5(b, a))
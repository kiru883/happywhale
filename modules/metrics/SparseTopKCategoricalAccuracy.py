import torch


def SparseTopKCategoricalAccuracy(predict, target, k):
    le = torch.argmax(target, dim=1)
    topk = torch.argsort(predict, dim=1, descending=True)[:, :k]

    match = (topk == le[..., None]).any(axis=1)

    return torch.sum(torch.sum(match)) / le.shape[0]


def SparseTop5CategoricalAccuracy(predict, target):
    return SparseTopKCategoricalAccuracy(predict, target, k=5)


def SparseTop1CategoricalAccuracy(predict, target):
    return SparseTopKCategoricalAccuracy(predict, target, k=1)




if __name__ == "__main__":
    a = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
    b = torch.tensor([[0.9, 0.05, 0.05], [0.01, 0.89, 0.1], [0.1, 0.01, 0.89], [0.1, 0.2, 0.6]])
    print(SparseTop1CategoricalAccuracy(b, a))

    #a = np.array([1, 2, 2, 3])
    #b = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 2], [3, 3, 3]])
    #print((b == a[..., None]).any(axis=1).astype(bool))



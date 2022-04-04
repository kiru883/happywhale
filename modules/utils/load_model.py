import torch


def load_model(model, model_path):
    state_dict = torch.load(model_path)['state_dict']
    state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model
import torch
import random
import os
import numpy as np
import pytorch_lightning as ptl


def SEED_EVERYTHING(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # !!!
    ptl.utilities.seed.seed_everything(seed=seed)







import os


def load_model():
    from network.Transformer import Transformer
    from cartoon import MODEL_ROOT
    import torch
    model = Transformer()
    model.load_state_dict(torch.load(os.path.join(MODEL_ROOT, "Hayao_net_G_float.pth")))
    model.eval()
    model.float()
    return model



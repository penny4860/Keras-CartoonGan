
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


if __name__ == '__main__':
    model = load_model()
    for i, p in enumerate(model.parameters()):
        param = p.data.numpy()
        print(i, param.shape)


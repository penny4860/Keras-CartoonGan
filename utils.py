# -*- coding: utf-8 -*-

import numpy as np
import os
from PIL import Image
from cartoon import SAMPLE_IMG

def load_net_in(img_fname=SAMPLE_IMG, desired_size=256):
    input_image = Image.open(img_fname).convert("RGB")
    input_image = input_image.resize((desired_size, desired_size), Image.BICUBIC)
    input_image = np.asarray(input_image)
    # RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]

    # preprocess, (-1, 1)
    input_image = input_image/255
    input_image = -1 + 2 * input_image
    return input_image

def run_by_torch(input_image):
    import torch
    from cartoon import MODEL_ROOT
    def load_model():
        model = Transformer()
        model.load_state_dict(torch.load(os.path.join(MODEL_ROOT, 'Hayao_net_G_float.pth')))
        model.eval()
        model.float()
        return model
    from torch.autograd import Variable
    from network.Transformer import Transformer
    import torchvision.transforms as transforms
    
    model = load_model()
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    input_image = Variable(input_image, volatile=True).float()
  
    # forward
    ys = model(input_image)
    ys = ys.data.cpu().float()
    ys = ys.numpy()
    ys = np.transpose(ys, [0, 2, 3, 1])
    return ys
    


import numpy as np
from PIL import Image

def load_net_in(img_fname, desired_size):
    input_image = Image.open(img_fname).convert("RGB")
    input_image = input_image.resize((desired_size, desired_size), Image.BICUBIC)
    input_image = np.asarray(input_image)
    # RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]

    # preprocess, (-1, 1)
    input_image = input_image/255
    input_image = -1 + 2 * input_image
    return input_image



import runway
import numpy as np
import argparse
import torch
from torchvision import transforms
import os.path


@runway.setup(options={'checkpoint_dir': runway.directory(description="runs folder"), 'checkpoint_dir2': runway.directory(description="pretrained weights") ,})
def setup(opts):
    path = Path(".")
    learn=load_learner(path, opts['checkpoint'])
    return learn


@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),}, outputs={'image': runway.image(description='output image containing the translated result')})
def translate(learn, inputs):
    img_fast = Image(img_t)
    p,img_hr,b = learn.predict(img_fast)
    return np.uint8(np.clip(image2np(img_hr), 0, 1)*255)




if __name__ == '__main__':
    runway.run(port=8889)

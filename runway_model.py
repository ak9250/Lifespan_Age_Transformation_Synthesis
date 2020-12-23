import runway
import numpy as np
import argparse
import torch
from torchvision import transforms
import os.path
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import util.preprocess_itw_im as preprocess
import util.deeplab as deepl

opt = TestOptions().parse(save=False)
opt.display_id = 0 # do not launch visdom
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.in_the_wild = True # This triggers preprocessing of in the wild images in the dataloader
opt.traverse = True # This tells the model to traverse the latent space between anchor classes
opt.interp_step = 0.05 # this controls the number of images to interpolate between anchor classes




@runway.setup(options={'checkpoint_dir': runway.directory(description="runs folder"), 'checkpoint_dir2': runway.directory(description="pretrained weights") ,'checkpoint_dir3': runway.file(extension='.dat',description="shape predictor") })
def setup(opts):
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    preprocess.resnet_file_path = opts['checkpoint_dir2'] + r'/R-101-GN-WS.pth.tar' 
    preprocess.deeplab_file_path = opts['checkpoint_dir2'] + r'/deeplab_model.pth'
    preprocess.predictor_file_path = opts['checkpoint_dir3'] 
    preprocess.model_fname = opts['checkpoint_dir2'] + r'/deeplab_model.pth'
    deepl.resnet101(opts['checkpoint_dir2'] + r'/R-101-GN-WS.pth.tar')
    opt.name = opts['checkpoint_dir']
    model = create_model(opt)
    return model


@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),}, outputs={'image': runway.image(description='output image containing the translated result')})
def translate(model, inputs):
    data = dataset.dataset.get_item_from_path(inputs['source_imgs'])
    visuals = model.inference(data)

    # os.makedirs('results', exist_ok=True)
    # out_path = os.path.join('results', os.path.splitext(img_path)[0].replace(' ', '_') + '.mp4')
    return visualizer.make_video(visuals, out_path)




if __name__ == '__main__':
    runway.run(port=8889)

import os
import numpy as np
import torch
import time
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.autograd import Variable
from pathlib import Path
from tqdm import tqdm

pix2pixhd_dir = Path('../src/pix2pixHD/')

import sys
sys.path.append(str(pix2pixhd_dir))

<<<<<<< HEAD
from data.fallon_data_loader import CreateDataLoader
=======
<<<<<<< HEAD
<<<<<<< HEAD
from data.fallon_data_loader import CreateDataLoader
=======
from data.canon_data_loader import CreateDataLoader
>>>>>>> 4363d0e... Fixed transfer so it displays canonical images and colored labels
=======
from data.data_loader import CreateDataLoader
>>>>>>> 52ff877... Normal transfer works fine
>>>>>>> 41ac1e5... Normal transfer works fine
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html

with open('../data/test_opt.pkl', mode='rb') as f:
    opt = pickle.load(f)

opt.dataroot = '/data/jl5/train_fallon/'
opt.verbose = True
opt.gpu_ids = [0]
opt.batchSize = 1
opt.checkpoints_dir = '../checkpoints'
opt.results_dir = "../results"

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

model = create_model(opt)

print(len(dataset))

for data in tqdm(dataset):
    minibatch = 1

    generated = model.inference(data['label'], data['inst'])
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    visualizer.save_images(webpage, visuals, img_path)
webpage.save()
torch.cuda.empty_cache()

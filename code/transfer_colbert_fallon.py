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
import ntpath

pix2pixhd_dir = Path('../src/pix2pixHD/')

import sys
sys.path.append(str(pix2pixhd_dir))

from data.canon_data_loader import CreateDataLoader
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
opt.label_nc = 0
opt.input_nc = 4
    
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

model = create_model(opt)


for data in tqdm(dataset):
	concat_list = data['label']
	canon_img = data['canon']

	for i in range(len(concat_list)):
		generated = model.inference(concat_list[i], data['inst'])
		img = canon_img[i]
		label = torch.narrow(concat_list[i][0], 0, 0, 1)
		visuals = OrderedDict([('input_label', util.tensor2label(label, 18)),
			('canonical_image', util.tensor2im(img[0])), ('synthesized_image', util.tensor2im(generated.data[0]))])
		base = ntpath.basename(data['path'][0])
		name = os.path.splitext(base)
		img_path = ["canon" + str(i) + "_" + name[0] + ".png"]
		visualizer.save_images(webpage, visuals, img_path)
webpage.save()
torch.cuda.empty_cache()





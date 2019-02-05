import numpy as np
import cv2
import torch
from tqdm import tqdm
from pathlib import Path

openpose_dir = Path('../src/pytorch_Realtime_Multi-Person_Pose_Estimation/')

import sys
sys.path.append(str(openpose_dir))
sys.path.append('../src/utils')


from network.rtpose_vgg import get_model
from evaluate.coco_eval import get_multiplier, get_outputs

# utils
from openpose_utils import remove_noise, get_pose

weight_name = openpose_dir.joinpath('network/weight/pose_model.pth')

# Get the vgg19 model from openpose, load weights, and make ready for cuda
model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

# Data Directory
save_dir = Path('/data/jl5/')
save_dir.mkdir(parents = True, exist_ok=True)

# Image Directory of Original Images
img_dir = save_dir.joinpath('fallondict')
img_dir.mkdir(parents = True, exist_ok=True)

# Traning Directiory
train_dir = save_dir.joinpath('train_fallon')
train_dir.mkdir(exist_ok=True)

# Create Training IMGs and Label Images of Poses
train_img_dir = train_dir.joinpath('train_img')
train_img_dir.mkdir(exist_ok=True)
train_label_dir = train_dir.joinpath('train_label')
train_label_dir.mkdir(exist_ok=True)

for idx in tqdm(range(1, 5404)):
	# get img_path
    img_path = img_dir.joinpath(f'frame{idx:06d}.jpg')
    img = cv2.imread(str(img_path))
    # grabs smallest img length
    shape_dst = np.min(img.shape[:2])
    # gets offset to create a square image
    oh = (img.shape[0] - shape_dst) // 2
    ow = (img.shape[1] - shape_dst) // 2
    # creates square image
    img = img[oh:oh+shape_dst, ow:ow+shape_dst]
    img = cv2.resize(img, (512, 512))

    # get the pose
    multiplier = get_multiplier(img)
    with torch.no_grad():
        paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
    r_heatmap = np.array([remove_noise(ht)
                      for ht in heatmap.transpose(2, 0, 1)[:-1]])\
                     .transpose(1, 2, 0)
    heatmap[:, :, :-1] = r_heatmap
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    print(heatmap.shape[:2])
    exit()
    label = get_pose(param, heatmap, paf)

    cv2.imwrite(str(train_img_dir.joinpath(f'img_{idx:06d}.png')), img)
    cv2.imwrite(str(train_label_dir.joinpath(f'label_{idx:06d}.png')), label)

torch.cuda.empty_cache()





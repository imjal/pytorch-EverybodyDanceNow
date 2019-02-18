import numpy as np
import cv2
import torch
from tqdm import tqdm
from pathlib import Path
import pdb
import pickle
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import matplotlib.patches as patches

openpose_dir = Path('../src/pytorch_Realtime_Multi-Person_Pose_Estimation/')

import sys
sys.path.append(str(openpose_dir))
sys.path.append('../src/utils')


from network.rtpose_vgg import get_model
from evaluate.coco_eval import get_multiplier, get_outputs

# utils
from openpose_utils import remove_noise, get_pose, get_pose_keypoints


def show_box_and_keypoints(im, box, x):

    fig,ax = plt.subplots(1)
    plt.scatter(x[:, 0], x[:, 1])
    ax.imshow(im)
    coord = current_box
    rect = patches.Rectangle((coord[0],coord[1]),coord[2], coord[3],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.savefig("bbox.png")
    plt.show()

def find_which_video(img_path):
        path = img_path[41:-4]
        index = int(path)
        ranges = {5114:0, 24213:1, 35465:2, 42253:3,
        57778:4, 64010:5, 72777:6, 81564:7, 94527:8, 102071:9}
        for i, canon_path in ranges.items():
            if index <= i:
                return canon_path
        return 10

weight_name = openpose_dir.joinpath('network/weight/pose_model.pth')

# Get the vgg19 model from openpose, load weights, and make ready for cuda]]
model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()


# Data Directory
img_dir = Path('/data/jl5/frames')

# make the directories for each video
video_dir = Path('/data/jl5/data-posewarp/train/frames/')
for i in range(11):
    new_dir = video_dir.joinpath(str(i))
    new_dir.mkdir(parents = True,  exist_ok = True)

ranges = [5114, 24213, 35465, 42253,
        57778, 64010, 72777, 81564, 94527, 102071, 101588]
k=0
for j in range(11):
    i = 1
    keypoints_arr = None
    boxes = None

    for idx in tqdm (range(k, ranges[j])):
        img_path = img_dir.joinpath(f'{idx:08d}.jpg')
        img = cv2.imread(str(img_path))
        if img is None:
            continue;

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
        keypoints = get_pose_keypoints(param, heatmap, paf)
        if (keypoints == -1).all():
            continue;

        if keypoints_arr is None:
          keypoints_arr = keypoints
        else:
          keypoints_arr = np.append(keypoints_arr, keypoints, axis = 2)
        current_box = [min(keypoints[:, 0]), min(keypoints[:, 1]), max(keypoints[:, 0])-min(keypoints[:, 0]), max(keypoints[:, 1])-min(keypoints[:, 1])]
        if boxes is None:
          boxes = current_box
        else:
          boxes = np.vstack((boxes, current_box))

        #show_box_and_keypoints(img, current_box, keypoints)
        new_dir = video_dir.joinpath(str(j))
        cv2.imwrite(str(new_dir.joinpath(f'{i:06d}.png')), img)
        i+=1
    with open("keypoints_vid" + str(j) + ".pkl", 'wb') as pickle_file:
        pickle.dump(keypoints_arr, pickle_file)
    with open("boxes_vid" + str(j) + ".pkl", 'wb') as pickle_file:
        pickle.dump(boxes, pickle_file)
    k = ranges[j]
torch.cuda.empty_cache()






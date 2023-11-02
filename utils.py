import torch
import math
import numpy as np
import torch
import nemo
from torch.utils import data
from DataReader import DataReader
from Dataset import Dataset
import numpy as np
import cv2

def rmse(input,predicted):
    return math.sqrt(((input - predicted) ** 2).sum()/input.shape[0])

def get_euclidean_err_2D(a, b):
    return math.sqrt(pow((a[0]-b[0]), 2) + pow((a[1]-b[1]), 2))

def get_euclidean_err_3D(a,b):
    return math.sqrt(pow((a[0] - b[0]), 2) + pow((a[1] - b[1]), 2) + pow((a[2] - b[2]), 2))

def test_loss_calc(loss_weighted_conf):
    test_conf = sum(loss_weighted_conf)/len(loss_weighted_conf)
    return test_conf

def get_fc_quantum(model):
    names = []
    # get all layers of the network
    for key, class_name in model.named_modules():
        names.append(key)
    # names[-1] is the last layer , use last layer name to get output eps      
    fc_quantum = model.relu4.get_output_eps(model.get_eps_at('relu4',1))
    return fc_quantum


def re_project_on_image(image_path,coordinates,img_size,out_path):
    image = cv2.imread(image_path)
    if img_size[0]<320 and img_size[1]<320:
        image = cv2.resize(image, (img_size[1],img_size[0]), interpolation=cv2.INTER_NEAREST)
    image_marked = cv2.circle(image, coordinates, radius=3, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(out_path,image_marked)


def resize_images(img_size,images_folder,out_folder):
    if img_size[0]<320 and img_size[1]<320:
        for count in range(1,101):
            image = cv2.imread(f"{images_folder}img_{count:06d}.png")
            image = cv2.resize(image, (img_size[1],img_size[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{out_folder}img_{count:06d}.png",image)
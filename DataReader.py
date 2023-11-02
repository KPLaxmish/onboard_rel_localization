import yaml
import os
import numpy as np
import cv2
import argparse
from PIL import Image
import cv2
import torch
from pathlib import Path

class DataReader:
    def process_train_data(repo,folder_path,yaml_path,img_size,stage):
        print("Loading yaml...")
        with open(yaml_path, 'r') as stream:
            synchronized_data = yaml.load(stream, Loader=yaml.CSafeLoader)
        print("Done")
                
        # access only the images, ignore camera calibration data, num_images = 10k
        train_img_names,train_imgs,train_labels = [],[],[]
        val_img_names,val_imgs, val_labels = [],[],[]
        
        # DEBUG: load all images for now
        all_images = []
        img_names = []
        all_labels = []
        for image_name, entry in synchronized_data.items():
            image_path = folder_path +image_name 
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #(320-H,320-W)
            if (img_size[0] < 320 and img_size[1]< 320):
                image = cv2.resize(image, (img_size[1],img_size[0]), interpolation=cv2.INTER_NEAREST)
            image = image[..., np.newaxis]# (320-H,320-W,1)
            image = image.astype('float32')
            image = np.swapaxes(image, 0, 2) # (1,320-W,320-H)
            image = np.swapaxes(image, 1, 2) # (1,320-H,320-W)
            all_images.append(image)
            img_names.append(image_name)
            train_x = [entry['pix'],entry['z-pos']]
            label_point = DataReader.preprocess_true_points(train_x,img_size=img_size) #(8000, 1, 320, 320)
            all_labels.append(label_point)
    
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)
        img_names = np.array(img_names)

        if(stage=='train'):
            # randomly decide which images end up in training,calibration and testing
            num = len(all_images)
            shuffled_idx =np.arange(num) # get an array [0, ..., num-1]
            np.random.shuffle(shuffled_idx) # in-place random shuffle of [0, ..., num-1]
            split_idx = int(num * 0.8) # decision at which index of the now shuffled shuffled_idx all_images and all_labels will be split

            train_imgs = all_images[shuffled_idx[:split_idx]] # only the images until the split index will be in the train set
            train_labels = all_labels[shuffled_idx[:split_idx]]
            train_img_names = img_names[shuffled_idx[:split_idx]]

            val_imgs = all_images[shuffled_idx[split_idx:]]  # only the images after split index will be in the test set
            val_labels = all_labels[shuffled_idx[split_idx:]]
            val_img_names = img_names[shuffled_idx[split_idx:]]

            calib_split = int(num * 0.05)
            calib_imgs = all_images[shuffled_idx[:calib_split]] # only the images until the split index will be in the calib set
            calib_labels = all_labels[shuffled_idx[:calib_split]]
            calib_img_names = img_names[shuffled_idx[:calib_split]]
            os.makedirs(repo+'outputs', exist_ok = True)
            np.save('outputs/calib_imgs', calib_imgs)
            np.save('outputs/calib_labels', calib_labels)
            np.save('outputs/calib_names', calib_img_names)
            return train_img_names,train_imgs, train_labels, val_img_names,val_imgs, val_labels
                
            
        elif(stage=='test'):
            return img_names,all_images, all_labels
        
        
    # preprocess true points to get 40x40x2 labels using  pix + z coordinates from pos
    def preprocess_true_points(point,img_size): # ------ change to points for multi robot case
            size = (int(img_size[0]/8),int(img_size[1]/8),2)
            label = np.zeros(size)
            #for point in points: --------- FOR MULTI ROBOT CASE, ignore for 1 robot case
            curW,curH = point[0]
            point_xy = np.array([int(curH*(img_size[0]/320)),int(curW*(img_size[1]/320))])
            point_z = point[1]
            xind, yind = (point_xy - 1) // 8
            #if xind != 0 and yind != 0:
            label[int(xind), int(yind), 0], label[int(xind), int(yind), 1] = 1.0, point_z
            # label[int(xind), int(yind), 0] = 1.0
            label_point = label
            label_point = label_point.astype('float32')
            label_point = np.swapaxes(label_point, 0, 2)  # <2, 40, 40>
            label_point = np.swapaxes(label_point, 1, 2)  # <2, 40, 40>
            return label_point
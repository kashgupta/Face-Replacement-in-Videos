'''
  File name: p3_dataloader.py
  Author: Kashish Gupta
  Date: 12-13-17
'''
import numpy as np
import PyNet
import cv2

def p3_dataloader():
    xy_labels = np.loadtxt('training.txt', usecols=range(1,11))
    name_labels = np.loadtxt('training.txt', usecols=range(1), dtype=str)

    all_labels_250 = []
    #all_labels_400 = []
    all_images_250 = []
    #all_images_400 = []
    for i in range(500): #name_labels.shape[0]
        print i
        name = name_labels[i]
        name = name.replace("\\", "/")
        label = xy_labels[i]
        label = np.array([label])
        #find the image
        img = cv2.imread(name)
        h, w, d = img.shape
        img[:, :, 0] = (img[:, :, 0] - 89.93) / 255
        img[:, :, 1] = (img[:, :, 1] - 99.50) / 255
        img[:, :, 2] = (img[:, :, 2] - 119.78) / 255
        gt_label = PyNet.get_gt_map(label, h, w)
        img_reshape = np.reshape(img, (3, h, w))

        gt_label_reshape = np.reshape(gt_label, (5,h,w))
        if h == 250:
            all_labels_250.append(gt_label_reshape)
            all_images_250.append(img_reshape)
        #elif h == 400:
            #all_labels_400.append(gt_label_reshape)
            #all_images_400.append(img_reshape)

    data_set_250 = PyNet.upsample2d(np.array(all_images_250), (40, 40))
    label_set_250 = PyNet.upsample2d(np.array(all_labels_250), (40, 40))
    #data_set_400 = PyNet.upsample2d(np.array(all_images_400), (40, 40))
    #label_set_400 = PyNet.upsample2d(np.array(all_labels_400), (40, 40))

    return [data_set_250, label_set_250]
    #return [data_set_250, data_set_400, label_set_250, label_set_400]
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob 
import os


def getLargestConnectedComponent(binary_img):
    '''
        Get the largest component in the Input Image
    '''
    binary_img = binary_img.astype(np.uint8)
    # -> Connected Components
    _, label_img = cv.connectedComponents(binary_img)
    # -> Count Number of pixel of each label
    labels, counts = np.unique(label_img, return_counts=True)

    # -> Remove Background
    counts = counts[labels!=0]
    labels = labels[labels!=0]

    # -> Get largest component
    largest_group_label = labels[np.argmax(counts)]
    output_img = np.zeros_like(binary_img, np.uint8)
    output_img[label_img==largest_group_label] = 1

    return output_img

def IoU(output_img, gt_img):
    '''
        Intersection Over Union
    '''
    intersect = np.logical_and(output_img, gt_img)
    union = np.logical_or(output_img, gt_img)
    iou = np.sum(intersect)/np.sum(union)
    return iou


if __name__ == "__main__":

    leaf_dir = "../Test_Dataset/Leaf Segmentation/Leaf/"
    groundtruth_dir = "../Test_Dataset/Leaf Segmentation/Groundtruth/"
    output_dir = "../Test_Dataset/Leaf Segmentation/Output/"

    os.makedirs(output_dir, exist_ok=True)

    leaf_path_list = sorted(glob(leaf_dir + "*"))
    groundtruth_path_list = sorted(glob(groundtruth_dir + "*"))

    for leaf_path, groundtruth_path in zip(leaf_path_list, groundtruth_path_list):
        leaf_img = cv.imread(leaf_path)
        groundtruth_img = cv.imread(groundtruth_path, 0)
        leaf_img_rgb = cv.cvtColor(leaf_img, cv.COLOR_BGR2RGB).astype(np.float64)

        green_minus_blue_img = leaf_img_rgb[:,:,1]-leaf_img_rgb[:,:,2]
        clip_img = np.clip(green_minus_blue_img, 0, 255).astype(np.uint8)
        _, seg_img = cv.threshold(clip_img, 10, 1, cv.THRESH_BINARY)

        largest_seg_img = getLargestConnectedComponent(seg_img)
        final_seg_img = largest_seg_img
        
        iou = IoU(final_seg_img, groundtruth_img)
        print(f"IoU = {iou:.3f}")

        # img_name = os.path.basename(leaf_path)
        # cv.imwrite(os.path.join(output_dir, img_name), final_seg_img*255)



import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm
import pyfeats

from skimage.measure import regionprops
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

def fillHoles(input_img):
    '''
        Fill Holes of a Binary Image
    '''
    # -> Create Buffer image
    buffer_img = np.zeros((input_img.shape[0]+2, input_img.shape[1]+2), np.uint8)
    buffer_img[1:-1, 1:-1] = input_img
    # -> Empty image
    empty_img = np.zeros((buffer_img.shape[0]+2, buffer_img.shape[1]+2), np.uint8)
    
    # -> Flood Fill
    _, flood_img, _, _ = cv.floodFill(buffer_img, empty_img, (0, 0), 1)
    flood_img = flood_img[1:-1, 1:-1]
    
    # -> Holes Masking
    hole_img = np.logical_not(flood_img)
    
    # -> Fill Holes
    output_img = np.logical_or(input_img, hole_img) 
    output_img = output_img.astype(np.uint8)

    return output_img

def getLargestConnectedComponent(binary_img):
    '''
        Remove Fragments in the Binary Image
    '''
    binary_img = binary_img.astype(np.uint8)
    # -> Connected Components
    _, label_img = cv.connectedComponents(binary_img)
    # -> Count Number of pixel of each label
    labels, counts = np.unique(label_img, return_counts=True)

    counts = counts[labels!=0]
    labels = labels[labels!=0]

    largest_group_label = labels[np.argmax(counts)]

    output_img = np.zeros_like(binary_img, np.uint8)
    output_img[label_img==largest_group_label] = 1

    return output_img

def regionBasedFeature(seg_img, input_img=None):

    _, label_img = cv.connectedComponents(seg_img)
    obj_list = regionprops(label_img, intensity_image=input_img)

    if input_img is not None:
        hsv_img = cv.cvtColor(input_img.astype(np.uint8), cv.COLOR_RGB2HSV)

    X = []
    for object in obj_list:

        y_indices, x_indices = np.where(label_img == object.label)

        h_mean, s_mean, v_mean = np.mean(hsv_img[y_indices,x_indices], axis=0)
        
        eccentricity = object.eccentricity

        f_list = [h_mean, s_mean, v_mean, eccentricity]

        X.append(f_list)
    
    return X


if __name__ == "__main__":

    nut_dataset_dir = "../Datasets/Nut Classification/"

    nut_classes = os.listdir(nut_dataset_dir)

    X = []
    y = []
    
    for class_label, nut_class in enumerate(tqdm(nut_classes)):
        nut_dir = os.path.join(nut_dataset_dir, nut_class)
        nut_path_list = glob(nut_dir + '/*')
    
        for path in nut_path_list:

            nut_img = cv.imread(path)
            nut_img_rgb = cv.cvtColor(nut_img, cv.COLOR_BGR2RGB).astype(np.float64)
            nut_img_gray = cv.cvtColor(nut_img, cv.COLOR_BGR2GRAY)

            ### -> Segmentation 
            _, seg_img = cv.threshold(nut_img_gray, None, 1, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            seg_img = fillHoles(seg_img)
            seg_img = getLargestConnectedComponent(seg_img)

            ### -> Feature Extraction

            # -> Region-based features
            region_feature = regionBasedFeature(seg_img, input_img=nut_img_rgb)
            # -> Texture-based features
            lte_feature, lte_labels = pyfeats.lte_measures(nut_img_gray, mask=seg_img, l=5)
            # -> Combine features
            f_vect = np.concatenate((region_feature[0], lte_feature))
    
            X.append(f_vect)
            y.append(class_label)

    ### -> Normalize features
    X_norm = normalize(X, norm="l2", axis=0)

    ### -> Split Train-test data
    X_train, X_test, y_train, y_true = train_test_split(X_norm, y, stratify=y, test_size=0.3, random_state=69)

    ### -> Classifier
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(X_train, y_train)

    y_pred = KNN.predict(X_test)
    print(classification_report(y_true, y_pred, target_names=nut_classes))

    conf_mat = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,0])
    print(conf_mat)
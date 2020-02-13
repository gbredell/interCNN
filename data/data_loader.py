__author__ = 'gbredell'
from torch.utils.data import Dataset
import numpy as np
import random
import scipy.ndimage
import torch
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

#### Define the data-augmentation functions
def min_max_normalization(image):
    '''
    :param image: Numpy image to perform max-min normalization on
    :return: Return max-min normalized numpy image
    '''
    max_pixel = np.max(image)
    min_pixel = np.min(image)
    diff = max_pixel - min_pixel

    image = ((image - min_pixel)/diff)
    return image

def aug_flip_vertical(image, truth):
    '''
    Perform a vertical flip of the numpy image with 50% probability
    :param image: The input image to network to be flipped (numpy)
    :param truth: The ground truth corresponding to the input image (numpy)
    :return: Flipped/Nor flipped image and ground truth (numpy)
    '''

    check = random.randint(0,1)
    if check < 1:
        return image, truth

    image = np.flip(image,1)
    truth = np.flip(truth,1)

    image = np.ascontiguousarray(image)
    truth = np.ascontiguousarray(truth)

    return image, truth

def aug_flip_horizontal(image, truth):
    '''
    Perform a horizontal flip of the numpy image with 50% probability
    :param image: The input image to network to be flipped (numpy)
    :param truth: The ground truth corresponding to the input image (numpy)
    :return: Flipped/Nor flipped image and ground truth (numpy)
    '''

    check = random.randint(0,1)
    if check < 1:
        return image, truth

    image = np.flip(image,0)
    truth = np.flip(truth,0)

    image = np.ascontiguousarray(image)
    truth = np.ascontiguousarray(truth)

    return image, truth

def aug_rotate(image, truth, angle):
    '''
    Rotates image and ground truth with an angel of "angel" with probability 50%
    :param image: Input image to be flipped (numpy)
    :param truth: Ground truth image corresponding to input image (numpy)
    :param angle: Angle to be flipped
    :return:
    '''

    check = random.randint(0,1)
    if check < 1:
        return image, truth

    degree = random.randint(-angle, angle)

    image = scipy.ndimage.interpolation.rotate(image, degree, reshape=False, order=1)
    truth = scipy.ndimage.interpolation.rotate(truth, degree, reshape=False, order=0)

    return image, truth

def aug_crop(image, truth, max_crop):
    '''
    Zoom into image with "max_crop" zoom with 50% probability while keeping image size by cropping
    :param image: Input image to be flipped (numpy)
    :param truth: Ground truth image corresponding to input image (numpy)
    :param max_crop: Zoom into image with this factor
    :return: Zoomed in image and ground truth in original sizes
    '''
    check = random.randint(0,1)
    if check < 1:
        return image, truth

    crop_size = random.uniform(1., max_crop)
    image_size = 320

    #Zoom the image, thus make it bigger
    image = scipy.ndimage.interpolation.zoom(image, crop_size, order=1)
    truth = scipy.ndimage.interpolation.zoom(truth, crop_size, order=0)

    #Now crop the image again
    x, y = image.shape
    startx = round(x/2)-round(image_size/2)
    starty = round(y/2)-round(image_size/2)
    endx = startx + image_size
    endy = starty + image_size

    image = image[startx:endx, starty:endy]
    truth = truth[startx:endx, starty:endy]

    return image, truth

def binary_converter(original_labels):
    '''
    Converts (NCI-ISBI 2013 challenge) mutliclass segmentation to binary, by converting pixels with
    value 2 to value 1
    :param original_labels: The data in torch format to be converted
    :return: Label image, where pixel values of 2 were converted to 1
    '''
    r , c = original_labels.shape
    twos_tensor = torch.ones(r,c).type(torch.LongTensor)*2
    original_labels = original_labels.type(torch.LongTensor)

    twos_mask = torch.eq(original_labels, twos_tensor).type(torch.LongTensor)
    subtract_mask = torch.ones(r,c).type(torch.LongTensor)*(-1)
    subtract_mask = torch.mul(subtract_mask, twos_mask)

    binary_label = torch.add(subtract_mask, original_labels)

    return binary_label

class DatasetCreater(Dataset):
    """
    autoCNN_img_0: Link to loaded numpy array of input images with shape [X_res, Y_res, N], where N is the number of slices
                   and X_res, Y_res the resolution of the images in each axis, respectively.
    autoCNN_labels_0: Link to loaded numpy array of segmented images with shape [X_res, Y_res, N]
    """

    # Initialize your data, by adding the links to the data files
    def __init__(self, transform, binary, train_img, train_seg):

        self.x_data = train_img
        self.y_data = train_seg

        self.len = train_img.shape[2]
        print("Number of slices in loader: ", self.len)

        self.transform = transform
        self.binary = binary

    def __getitem__(self, index):
        image = self.x_data[:,:,index]
        seg = self.y_data[:,:,index]

        if self.transform:
            image, seg = aug_flip_vertical(image, seg)
            image, seg = aug_flip_horizontal(image, seg)
            image, seg = aug_rotate(image, seg, 15)
            image, seg = aug_crop(image, seg, 1.3)

        image = min_max_normalization(image)

        image = torch.from_numpy(image)
        seg = torch.from_numpy(seg)

        if self.binary:
            seg = binary_converter(seg)

        return image, seg

    def __len__(self):
        return self.len

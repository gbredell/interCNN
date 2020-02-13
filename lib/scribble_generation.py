__author__ = 'gbredell'
import torch
import numpy as np
import random

def incorrect_pixels(prediction, labels_user_model):
    '''
    Identifies incorrectly labeled pixels for all three classes (ex. which pixels should have been 0, but were another class)
    :param prediction: Prediction of interCNN
    :param labels_user_model: Corresponding ground truth segmentation
    :return: Label map and incorrectly classified pixels for the classes 0, 1, 2
    '''

    #Take difference between labels and prediction --> correctly labeled = 0
    diff_label_pred = prediction - labels_user_model

    #Make all the correctly labeled data 1 and all other data 0
    b, r , c = diff_label_pred.shape
    zeros_tensor = torch.zeros(b,r,c).type(torch.LongTensor)
    correct_pred = torch.eq(zeros_tensor, diff_label_pred)

    #Make all the correctly labeled data 0 and all the incorrectly labeled data 1
    correct_pred = correct_pred.type(torch.LongTensor)
    ones_tensor = torch.ones(b,r,c).type(torch.LongTensor)
    incorrect_pred = torch.eq(zeros_tensor, correct_pred)

    #Next we will first extract all the pixels of a specific class from the labels and then multiply with the incorrect mask
    #Case background: 0
    label_0 = torch.eq(zeros_tensor, labels_user_model)
    incorrect_0 = torch.mul(label_0, incorrect_pred)

    #Case peripheral: 1
    label_1 = torch.eq(ones_tensor, labels_user_model)
    incorrect_1 = torch.mul(label_1, incorrect_pred)

    #Case central: 2
    twos_tensor = ones_tensor * 2
    label_2 = torch.eq(twos_tensor, labels_user_model)
    incorrect_2 = torch.mul(label_2, incorrect_pred)

    return(incorrect_0, incorrect_1, incorrect_2, label_0, label_1, label_2)

#Block mask maker specific to 320x320 resolution
def mask_checker(number, pixel_random, x_res, y_res):
    '''
    Makes a box scribble around the center pixel of size number
    :param number: How large the box around the center pixel should be
    :param pixel_random: The pixel around which the box should be placed
    :param x_res: Resolution in x-direction
    :param y_res: Resolution in y-direction
    :return: List of pixels that are chosen as scribbles
    '''

    pixel_list = np.array([])
    total_length = x_res*y_res
    up = x_res
    down = -x_res
    left = -1
    right = 1

    pixel_list = np.append(pixel_list, [pixel_random])

    for i in range(0, number):
        for q in range(0, number):
            if i == 0:
                #The right upper box
                pixel_select = pixel_random + right*q
                if (pixel_select >= 0) & (pixel_select < total_length):
                    if pixel_select != pixel_random:
                        pixel_list = np.append(pixel_list, [pixel_select])
                #The left lower box
                pixel_select = pixel_random + left*q
                if (pixel_select >= 0) & (pixel_select < total_length):
                    if pixel_select != pixel_random:
                        pixel_list = np.append(pixel_list, [pixel_select])

            elif q == 0:
                #The right upper box
                pixel_select = pixel_random + up*i
                if (pixel_select >= 0) & (pixel_select < total_length):
                    if pixel_select != pixel_random:
                        pixel_list = np.append(pixel_list, [pixel_select])
                #The left lower box
                pixel_select = pixel_random + down*i
                if (pixel_select >= 0) & (pixel_select < total_length):
                    if pixel_select != pixel_random:
                        pixel_list = np.append(pixel_list, [pixel_select])

            else:
                #The right upper box
                pixel_select = pixel_random + up*i + right*q
                if (pixel_select >= 0) & (pixel_select < total_length):
                    if pixel_select != pixel_random:
                        pixel_list = np.append(pixel_list, [pixel_select])
                #The left upper box
                pixel_select = pixel_random + up*i + left*q
                if (pixel_select >= 0) & (pixel_select < total_length):
                    if pixel_select != pixel_random:
                        pixel_list = np.append(pixel_list, [pixel_select])
                #The left lower box
                pixel_select = pixel_random + down*i + left*q
                if (pixel_select >= 0) & (pixel_select < total_length):
                    if pixel_select != pixel_random:
                        pixel_list = np.append(pixel_list, [pixel_select])
                #The right lower box
                pixel_select = pixel_random + down*i + right*q
                if (pixel_select >= 0) & (pixel_select < total_length):
                    if pixel_select != pixel_random:
                        pixel_list = np.append(pixel_list, [pixel_select])

    length = len(pixel_list)

    return pixel_list.astype(int), length

def mask_creater(incorrect, label, class_number, scribbles_array):
    '''
    Scribble generating function.
    :param incorrect: Maps with the location of the pixels that have the wrong class
    :param label: Maps of the correct class location for each pixel
    :param class_number: For which class scribbles should be generated (0,1 or 2)
    :param scribbles_array: The array of previous scribbles to which the new scribbles should be added to
    :return: New scribble map for the specific class, combining the old scribbles with the newly generated ones
    '''

    #Reshape tensor to array with all the pixels in one column
    c, h, w = incorrect.size()
    incorrect = incorrect.permute(1,2,0)
    incorrect = incorrect.contiguous().view(-1, c)

    c, h, w = label.size()
    label = label.permute(1,2,0)
    label = label.contiguous().view(-1, c)

    #Get all the pixels in one row that is wrongly classified and select one random pixel
    for p in range(0,c):

        label_p = label[:,p]
        length_label = len(label_p)
        incorrect_p = incorrect[:,p]
        scribble_p = scribbles_array[p,:,:]
        scribble_p = scribble_p.flatten()

        #Make sure label is present in the image. Otherwise scribbles not necessary and all 5
        if length_label > 0:
            incorrect_p = incorrect_p.numpy()
            incorrect_pixel_locations, = np.where(incorrect_p == 1)
            length = len(incorrect_pixel_locations)

            if length > 20:
                pixel_position = random.randint(0,length-1)
                pixel_random = incorrect_pixel_locations[pixel_position].item()

                #Select pixels around the randomly selected pixel
                list_0, length = mask_checker(4, pixel_random, w, h)

                #Add these selected pixels to the scribble
                scribble_p[list_0] = class_number
                scribble_p =scribble_p.reshape(320,320)
                scribbles_array[p,:,:] = scribble_p

    return scribbles_array

def scribble_input(pred, labels, scrib_prev = 0, initial = False):

    incorrect_0, incorrect_1, incorrect_2, label_0, label_1, label_2 = incorrect_pixels(pred, labels)

    if initial:
        c, h, w = incorrect_0.size()
        scribbles_initial = np.ones((c, h, w))*5
        scribbles_0 = mask_creater(incorrect_0, label_0, 0, scribbles_initial)
    else:
        scrib_previous = scrib_prev.cpu().data.numpy().squeeze(1)
        scribbles_0 = mask_creater(incorrect_0, label_0, 0, scrib_previous)

    scribbles_1 = mask_creater(incorrect_1, label_1, 1, scribbles_0)
    scribbles_2 = mask_creater(incorrect_2, label_2, 2, scribbles_1)

    return torch.from_numpy(scribbles_2).unsqueeze(1).float().cuda()

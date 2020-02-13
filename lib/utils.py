__author__ = 'gbredell'

import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

# ### Define the dice-score evaluator

def class_dice(pred, target, class_num):

    #get the size of the tensor and make tensor filled with the class number of the same lenght
    length = len(pred)
    number = torch.LongTensor([class_num]*length)

    #Look where does the label have pixels of that class and count them
    target_ones = torch.eq(number, target)
    target_ones_sum = torch.sum(target_ones.float())

    #Look where does the prediction have pixels of that class and count them
    pred_ones = torch.eq(number, pred)
    pred_ones_sum = torch.sum(pred_ones.float())

    #See if intersection = 0, because label not present in both prediction & label
    if (pred_ones_sum == 0) & (target_ones_sum == 0):
        accuracy = 1

    #See if there are this class in the image. Can't feed this case to f1_score, because ill defined
    elif (pred_ones_sum != 0) & (target_ones_sum == 0):
        accuracy = 0

    #Calculate the dice score if this is not the case
    else:
        labels_include = [class_num]
        accuracy = f1_score(target, pred, labels_include, average=None)
        accuracy = accuracy[0]

    return accuracy

def cross_entropy2d(input, target, weight=None, size_average=False):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

# Function that converts output of interCNN to a format that can be inputed into the interCNN again
def prediction_converter(outputs):
    softmax = nn.Softmax2d()
    maxIn = softmax(outputs)
    maxIn = torch.max(maxIn, 1)[1]
    prediction = maxIn.data.cpu()

    #Output is the prediciton as LonTensor [Batch Size, 320, 320]
    return prediction
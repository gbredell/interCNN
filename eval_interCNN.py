__author__ = 'gbredell'
import NCI_ISBI_2013 as nci
import data_loader as dl
import config
import models
import torch
import eval_func as val

binary = False
if binary:
    num_classes = 2
else:
    num_classes = 3

#Import data
test_dataset = dl.DatasetCreater(False, binary, nci.test_img, nci.test_seg)
test_loader_interCNN = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

#Import models
cnn1 = models.autoCNN(num_classes).cuda()
cnn1.load_state_dict(torch.load(config.autoCNN_pth))
cnn1.eval();

cnn2 = models.interCNN(num_classes).cuda()
cnn2.load_state_dict(torch.load(config.interCNN_pth))
cnn2.eval();

#Evaluate interCNN
val.interCNN_test(cnn1, cnn2, test_loader_interCNN, val_iterations = 20, controls = 3, num_class = num_classes)
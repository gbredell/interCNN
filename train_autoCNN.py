__author__ = 'gbredell'

from data import NCI_ISBI_2013 as nci
from data import data_loader as dl
import numpy as np
from config import paths
from lib import models
from lib import utils
import torch
from lib import eval_func as val
from lib import itertools as it

learning_rate = 0.0001
num_epochs_autoCNN = 400
batch_size = 4
val_internal = 20
binary = False

if binary:
    num_classes = 2
else:
    num_classes = 3

print(num_classes)

#Import the data
training_dataset_autoCNN = dl.DatasetCreater(True, binary, nci.autoCNN_train_img, nci.autoCNN_train_seg)
val_dataset_autoCNN = dl.DatasetCreater(False, binary, nci.autoCNN_val_img, nci.autoCNN_val_seg)

train_loader_autoCNN = torch.utils.data.DataLoader(dataset=training_dataset_autoCNN, batch_size=batch_size, num_workers=4, shuffle=True)
val_loader_autoCNN = torch.utils.data.DataLoader(dataset=val_dataset_autoCNN, batch_size=1, shuffle=False)

#Import the model
cnn1 = models.autoCNN(num_classes).cuda()

#Define optimizer
optimizer = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)

# Train the Model
loss_list = []
it_num = 0

for epoch in range(num_epochs_autoCNN):
    for i, (images, labels) in enumerate(train_loader_autoCNN):
        #Increase the iteration number:
        it_num = it_num + 1

        images = images.unsqueeze(1)
        images = images.float().cuda()
        labels = labels.type(torch.LongTensor).cuda()

        #Prediction from CNN1
        optimizer.zero_grad()
        outputs = cnn1(images)
        loss = utils.cross_entropy2d(input = outputs, target = labels, weight=None, size_average=False)
        loss.backward()
        optimizer.step()

        if it_num%val_internal == 0:
            #Validation score tracker
            cnn1_dc = val.autoCNN_test(cnn1, val_loader_autoCNN, num_class = num_classes, checker = True)
            if it_num == val_internal:
                class_cnn1_score = cnn1_dc
            else:
                class_cnn1_score = np.concatenate((class_cnn1_score ,cnn1_dc), axis = 0)

            loss_list = np.append(loss_list, loss.data.cpu())
            print("Epoch Number: ", epoch, '/', num_epochs_autoCNN, " Dice Score: ", cnn1_dc[0,:].flatten(), " Loss: ", loss.data.cpu())

            #Save the parameters
            np.save(paths.save_val_pth + 'autoCNN_class_score.npy', class_cnn1_score)
            torch.save(cnn1.state_dict(), paths.save_model_pth + 'autoCNN_last.pt')

            if len(loss_list) > 50:
                #Save the cnn with the best validation score out of the average
                if (np.mean(class_cnn1_score[-51:-1, :])) < (np.mean(class_cnn1_score[-50:, :])):
                    torch.save(cnn1.state_dict(), paths.save_model_pth + 'autoCNN_best.pt')

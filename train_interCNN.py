__author__ = 'gbredell'
import NCI_ISBI_2013 as nci
import data_loader as dl
import numpy as np
import config
import models
import utils
import scribble_generation as sg
import torch
import eval_func as val

learning_rate = 0.0001
num_epochs_interCNN = 80
batch_size = 4
val_internal = 10
max_iterations = 11
binary = False

if binary:
    num_classes = 2
else:
    num_classes = 3

print(num_classes)

#Import the data
training_dataset_interCNN = dl.DatasetCreater(True, binary, nci.interCNN_train_img, nci.interCNN_train_seg)
val_dataset_interCNN = dl.DatasetCreater(False, binary, nci.interCNN_val_img, nci.interCNN_val_seg)

train_loader_interCNN = torch.utils.data.DataLoader(dataset=training_dataset_interCNN, batch_size=batch_size, num_workers=4, shuffle=True)
val_loader_interCNN = torch.utils.data.DataLoader(dataset=val_dataset_interCNN, batch_size=1, shuffle=False)

#Import the model
cnn1 = models.autoCNN(num_classes).cuda()
cnn1.load_state_dict(torch.load(config.autoCNN_pth))
cnn1.eval();

cnn2 = models.interCNN(num_classes).cuda()

#Define optimizer
optimizer_2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

# Train the Model
loss_list = []
it_num = 0

for epoch in range(num_epochs_interCNN):
    for i, (images, labels) in enumerate(train_loader_interCNN):
        #Increase the iteration number:
        it_num = it_num + 1

        images = images.unsqueeze(1)
        images = images.float().cuda()
        labels_user_model = labels.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor).cuda()

        #Prediction from CNN1
        outputs = cnn1(images)

        #Transform output to correct size and format
        prediction = utils.prediction_converter(outputs)

        #Use image, prediction and scribbles as new input
        scribbles = sg.scribble_input(prediction, labels_user_model, initial = True)
        prediction = prediction.unsqueeze(1).float().cuda()

        for i in range(1, max_iterations):
            optimizer_2.zero_grad()
            outputs = cnn2(images, prediction, scribbles)

            loss = utils.cross_entropy2d(input = outputs, target = labels, weight=None, size_average=False)

            loss.backward()
            optimizer_2.step()

            #Transform output to correct size and format
            prediction = utils.prediction_converter(outputs)

            #Use image, prediction and scribbles as new input
            scribbles = sg.scribble_input(prediction, labels_user_model, scribbles)
            prediction = prediction.unsqueeze(1).float().cuda()

        if it_num%val_internal == 0:
            #Validation score tracker
            cnn1_dc, cnn2_dc = val.interCNN_test(cnn1, cnn2, val_loader_interCNN, num_class = num_classes, checker = True)
            if it_num == val_internal:
                class_cnn1_score = cnn1_dc
                class_cnn2_score = cnn2_dc
            else:
                class_cnn1_score = np.concatenate((class_cnn1_score ,cnn1_dc), axis = 0)
                class_cnn2_score = np.concatenate((class_cnn2_score ,cnn2_dc), axis = 0)

            loss_list = np.append(loss_list, loss.data.cpu())
            print("Epoch Number: ", epoch, '/', num_epochs_interCNN, " Dice Score: ", np.mean(cnn2_dc, axis = 2).flatten(), " Loss: ", loss.data.cpu())

            #Save the parameters
            np.save(config.save_val_pth + 'class_cnn1_score.npy', class_cnn1_score)
            np.save(config.save_val_pth + 'class_cnn2_score.npy', class_cnn2_score)
            torch.save(cnn2.state_dict(), config.save_model_pth + 'interCNN_last_xxx.pt')

            if len(loss_list) > 51:
                #Save the cnn with the best validation score out of the average
                if np.mean(class_cnn2_score[-51:-1,:]) < np.mean(class_cnn2_score[-50:, :]):
                    torch.save(cnn2.state_dict(), config.save_model_pth + 'interCNN_best_xxx.pt')

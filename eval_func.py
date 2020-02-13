__author__ = 'gbredell'
import torch
import scribble_generation as sg
import utils
import numpy as np
import config

def interCNN_test(cnn1, cnn_saved, loader_data, val_iterations = 10, controls = 1, num_class = 2,  save = True, checker = False):

    cnn1.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    cnn_saved.eval()
    total = 0

    avg_class_cnn1 = np.zeros(((num_class-1),1))
    avg_class_cnn2 = np.zeros(((num_class-1), controls, val_iterations))

    for m in range(0,controls):
        for images, labels in loader_data:
            total = total + 1
            images = images.unsqueeze(1).float().cuda()
            labels = labels.type(torch.LongTensor)
            labels_user_model = labels
            labels = labels.cuda()

            # Predict
            outputs = cnn1(images)

            #Convert prediction and labels
            prediction_cnn1 = utils.prediction_converter(outputs)
            prediction_cnn1_no_transform = prediction_cnn1
            c, h, w = prediction_cnn1.size()
            prediction_cnn1 = prediction_cnn1.permute(1,2,0).contiguous().view(-1, c).squeeze(1)
            labels = labels.permute(1,2,0).contiguous().view(-1, c).squeeze(1)

            #Calculate class value for cnn1
            for q in range(0, (num_class-1)):
                avg_class_cnn1[q,:] = avg_class_cnn1[q,:] + utils.class_dice(prediction_cnn1, labels.data.cpu(), q+1)

            #Use image, prediction and scribbles as new input
            scribbles = sg.scribble_input(prediction_cnn1_no_transform, labels_user_model, initial = True)
            prediction = prediction_cnn1_no_transform.unsqueeze(1).float().cuda()

            for i in range(0,val_iterations):
                #Make new prediction
                outputs = cnn_saved(images, prediction, scribbles)

                #Convert prediction to format for dice_score
                new_prediction = utils.prediction_converter(outputs)
                new_prediction_no_transform = new_prediction
                c, h, w = new_prediction.size()
                new_prediction = new_prediction.contiguous().view(-1, c).squeeze(1)

                #Compute the dice score
                for q in range(0, (num_class-1)):
                    avg_class_cnn2[q,m,i] = avg_class_cnn2[q,m,i] + utils.class_dice(new_prediction, labels.data.cpu(), q+1)

                #Use image, prediction and scribbles as new input
                scribbles = sg.scribble_input(new_prediction_no_transform, labels_user_model, scribbles)
                prediction = new_prediction_no_transform.unsqueeze(1).float().cuda()

    if save:
        #Save the parameters
        np.save(config.save_test_pth + 'avg_class_cnn1.npy', avg_class_cnn1/total)
        np.save(config.save_test_pth + 'avg_class_cnn2.npy', np.mean(avg_class_cnn2, axis=1)/(total/controls))
        np.save(config.save_test_pth + 'std_class_cnn2.npy', np.std(avg_class_cnn2/(total/controls), axis=1))

    if checker:
        return np.expand_dims(avg_class_cnn1/total, axis=0), np.expand_dims(np.mean(avg_class_cnn2, axis=1)/(total/controls), axis=0)
    else:
        return


def autoCNN_test(cnn1, loader_data, controls = 1, num_class = 2,  save = True, checker = False):

    cnn1.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0
    avg_class_cnn1 = np.zeros(((num_class-1),1))

    for m in range(0,controls):
        for images, labels in loader_data:
            total = total + 1
            images = images.unsqueeze(1).float().cuda()
            labels = labels.type(torch.LongTensor)
            labels = labels.cuda()

            # Predict
            outputs = cnn1(images)

            #Convert prediction and labels
            prediction_cnn1 = utils.prediction_converter(outputs)
            c, h, w = prediction_cnn1.size()
            prediction_cnn1 = prediction_cnn1.permute(1,2,0).contiguous().view(-1, c).squeeze(1)
            labels = labels.permute(1,2,0).contiguous().view(-1, c).squeeze(1)

            #Calculate class value for cnn1
            for q in range(0, (num_class-1)):
                avg_class_cnn1[q,:] = avg_class_cnn1[q,:] + utils.class_dice(prediction_cnn1, labels.data.cpu(), q+1)

    if save:
        #Save the parameters
        np.save(config.save_test_pth + 'avg_class_cnn1.npy', avg_class_cnn1/total)

    if checker:
        return np.expand_dims(avg_class_cnn1/total, axis=0)
    else:
        return


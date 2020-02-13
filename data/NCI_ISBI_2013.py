__author__ = 'gbredell'
import data_loader
from config import paths
import numpy as np
import os
import pydicom as dicom
import nrrd

def load_prostate_img_and_labels(train_ids_list = paths.data_split ,data_path_tr = paths.input_pth ,seg_path_tr = paths.seg_pth):
    '''
    Loads the dicom files and put all of the medical images and corresponding segmentations in numpy arrays
    :param train_ids_list: Link to numpy file specifying how the data should be split
    :param data_path_tr: Link to file with the medical images (DICOM)
    :param seg_path_tr: Link to the file with segmentations
    :return: Numpy array for medical images and segmentations in shape [X_res, Y_res, N], where N is the number of slices
    '''
    count=0
    for study_id in train_ids_list:
        PathDicom = str(data_path_tr)+str(study_id)

        lstFilesDCM = []  # create an empty list
        for dirName, subdirList, fileList in os.walk(PathDicom):
            fileList.sort()
            for filename in fileList:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    lstFilesDCM.append(os.path.join(dirName,filename))

        # Get ref file
        RefDs = dicom.read_file(lstFilesDCM[0])

        # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

        # Load spacing values (in mm)
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

        # The array is sized based on 'ConstPixelDims'
        img = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

        # loop through all the DICOM files
        for filenameDCM in lstFilesDCM:
            # read the file
            ds = dicom.read_file(filenameDCM)
            # store the raw image data
            img[:, :, ds.InstanceNumber-1] = (ds.pixel_array)

        PathSeg = str(seg_path_tr)+str(study_id)+".nrrd"
        seg, options = nrrd.read(PathSeg)

        #fix swap axis
        seg=np.swapaxes(seg,0,1)

        if(count==0):
            img_final=img
            seg_final=seg
            count=count+1
        else:
            img_final=np.concatenate((img_final,img),axis=2)
            seg_final=np.concatenate((seg_final,seg),axis=2)
            count=count+1

    return img_final.astype(np.int16),seg_final

def data_selection(select_list):
    '''
    Loading selected patient files into a numpy array
    :param select_list: List of patients IDs to load into the array
    :return: Input data and segmentation numpy array of selected patients [X_res, Y_res, N]
    '''

    for i in range (0,len(select_list)):
        case_number = select_list[i]

        if case_number < 10:
            string = ["000" + str(case_number)]
        if case_number >= 10:
            string = ["00" + str(case_number)]

        select_img, select_labels = load_prostate_img_and_labels(train_ids_list = string)

        if i == 0:
            select_img_con = select_img
            select_labels_con = select_labels

        else:
            select_img_con = np.concatenate((select_img_con, select_img),2)
            select_labels_con = np.concatenate((select_labels_con, select_labels),2)

    return select_img_con, select_labels_con

#list_train: Used for autoCNN training
#list_train2: New data used for interCNN training
#list_train_combo: Total data used to train interCNN
#list_val_training: Patient case used for validation (to stop training)
#list_test: Patient cases used for testing

list_total = np.load(paths.data_split)

data_splitter = {'autoCNN_train': list_total[:15], 'autoCNN_val': list_total[15:23], 'interCNN_train': list_total[:23],
                    'interCNN_val': [list_total[23]], 'test': list_total[24:29]}

autoCNN_train_img, autoCNN_train_seg = data_selection(data_splitter['autoCNN_train'])
autoCNN_val_img, autoCNN_val_seg = data_selection(data_splitter['autoCNN_val'])
interCNN_train_img, interCNN_train_seg = data_selection(data_splitter['interCNN_train'])
interCNN_val_img, interCNN_val_seg = data_selection(data_splitter['interCNN_val'])
test_img, test_seg = data_selection(data_splitter['test'])



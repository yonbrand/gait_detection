# ## Gait Segmentation
# This code aims to take an acceleration raw data from daily-living, and extract the gait (walking) episodes during the recording period.
# This code is based on the original paper code: https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/code/gait-extraction/tf_seg_new.ipynb
#
# Converted from TensorFlow to Pytorch



import os
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pickle
import random
# import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeaveOneOut
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as sio
import scipy
from scipy import signal
from sklearn.metrics import precision_recall_curve
import re
import io
from visdom import Visdom
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000000000
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from ismember import ismember

from scipy.stats import pearsonr
from scipy.stats import spearmanr

#
# import sklearn
# from sklearn.metrics import roc_auc_score
# import ignite.distributed as idist
# from ignite.contrib.metrics import ROC_AUC
# from ignite.engine import Engine
# from ignite.exceptions import NotComputableError
# from ignite.metrics.epoch_metric import EpochMetricWarning

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# # Load & process the data
def read_data(data_path,fs):
    file_names = os.listdir(data_path)
    n_counts = int(20.48*fs)
    print('dont forget to set manually the last layer of each network to n_counts (512)')
    all_data = np.empty((0, 1, 3, n_counts))
    for ind in file_names:
        curr_file = str(data_path + '/' + ind)
        acc_dict = sio.loadmat(curr_file)
        acc_np = acc_dict[list(acc_dict.keys())[-1]] #acc_dict['acc'] if not resampled
        data_n = acc_np[0:((len(acc_np) // n_counts) * n_counts)]  # remove part of the data for dividing without residual
        sample_data = data_n.reshape((len(data_n) // n_counts), 1, 3,n_counts)  # divide the data to samples in length of 512
        all_data = np.append(all_data, sample_data, axis=0)
    return all_data

def read_labels(label_path,fs):
    file_names = os.listdir(label_path)
    n_counts=int(20.48*fs)
    all_labels = np.empty((0, n_counts))
    all_participants = np.empty((0, n_counts))
    count = 1
    for ind in file_names:
        curr_file = str(label_path + '/' + ind)
        labels_dict = sio.loadmat(curr_file)
        labels_np=labels_dict[list(labels_dict.keys())[-1]] #resampled labels
        labels_n = labels_np[0:((len(labels_np) // n_counts) * n_counts)]
        sample_labels = labels_n.reshape((len(labels_n) // n_counts), n_counts)
        all_labels = np.append(all_labels, sample_labels, axis=0)
        participant_vec = np.full_like(sample_labels, count)
        all_participants = np.append(all_participants, participant_vec, axis=0)
        count += 1
    return all_labels, all_participants

def read_data_labels(data_path,label_path,data_type,fs,std_thresh, Resample, res_rate, comp_analysis, Filtering):
    file_names = os.listdir(data_path)
    n_counts = int(512 * res_rate)
    print('dont forget to set manually the last layer of each network to ' + str(n_counts))
    all_labels = np.empty((0, n_counts))
    all_participants = np.empty((0, n_counts))
    ind_potential_gait=np.empty((0,))
    all_data = np.empty((0, 1, 3, n_counts))
    count=1
    for ind in range(len(file_names)):
        curr_data = str(data_path + '/acc'+data_type+str(ind+1))
        acc_dict = sio.loadmat(curr_data)
        acc= acc_dict['acc'] #acc_dict['acc'] if not resampled
        curr_labels = str(label_path + '/labels'+data_type + str(ind+1))
        labels_dict = sio.loadmat(curr_labels)
        labels= labels_dict['labels']
        data_new, labels_new, ind_potential_gait_std=preprocessing(acc, labels,fs, std_thresh, Resample, res_rate, comp_analysis, Filtering)
        ind_potential_gait=np.append(ind_potential_gait,ind_potential_gait_std) #save the indexes of the part of gait that removed in the moving std process
        data_n = data_new[:,0:((data_new.shape[1] // n_counts) * n_counts)]  # remove part of the data for dividing without residual
        sample_data = data_n.reshape((data_n.shape[1] // n_counts), 1, 3,n_counts)  # divide the data to samples in length of 512
        all_data = np.append(all_data, sample_data, axis=0)
        labels_n = labels_new[0:((len(labels_new) // n_counts) * n_counts)]
        sample_labels = labels_n.reshape((len(labels_n) // n_counts), n_counts)
        all_labels = np.append(all_labels, sample_labels, axis=0)
        participant_vec = np.full_like(sample_labels, count)
        all_participants = np.append(all_participants, participant_vec, axis=0)
        count += 1
    return all_data, all_labels,all_participants,ind_potential_gait

def read_physionet(main_path, down_sample):
    file_names = os.listdir(main_path+ '/raw_accelerometry_data')
    all_data = np.empty((0, 1, 3, 512))
    all_labels = np.empty((0, 512))
    all_participants = np.empty((0, 512))
    count = 1

    for ind in file_names:
        curr_file = str(main_path + '/raw_accelerometry_data/' + ind)
        print(curr_file)
        df = pd.read_csv(os.path.join(curr_file))
        print('df')
        n_counts = 512
        data = df.iloc[0:((len(df) // n_counts) * n_counts)]  # remove part of the data for dividing without residual
        acc = data[['lw_x', 'lw_y', 'lw_z']]
        acc_np = acc.to_numpy()
        labels = data['activity']
        labels[data['activity'] != 1] = 0
        labels_np = labels.to_numpy()

        # downsample from 100Hz to 25Hz
        if down_sample=="down_sample":
            downSampleIndex = np.linspace(0, len(acc_np), int(1 + np.floor(len(acc_np) / 4)))
            downSampleIndex = downSampleIndex.astype(int)
            downSampleIndex = downSampleIndex[:-1]
            acc_np = np.asarray([np.take(acc_np[:,i], downSampleIndex) for i in range(acc_np.shape[1])])
            acc_np= acc_np.transpose()

            labels_np = np.take(labels_np, downSampleIndex)
        acc_np=acc_np[0: ((max(acc_np.shape) // n_counts) * n_counts),:]
        labels_np = labels_np[0: ((max(labels_np.shape) // n_counts) * n_counts)]
        sample_data = acc_np.reshape((max(acc_np.shape) // n_counts), 1, 3, n_counts)
        all_data = np.append(all_data, sample_data, axis=0)
        sample_labels = labels_np.reshape((len(labels_np) // 512), 512)
        all_labels = np.append(all_labels, sample_labels, axis=0)
        participant_vec = np.full_like(sample_labels, count)
        all_participants = np.append(all_participants, participant_vec, axis=0)
        count += 1

    with open(main_path + '/data_physionet_'+down_sample+'.npy', 'wb') as f:
        pickle.dump(all_data, f, protocol=4)
    with open(main_path + '/labels_physionet_'+down_sample+'.npy', 'wb') as f:
        pickle.dump(all_labels, f, protocol=4)
    with open(main_path + '/participants_physionet_'+down_sample+'.npy', 'wb') as f:
        pickle.dump(all_participants, f, protocol=4)

def data_processing(main_path,file_version,data_type,data, labels, batch_size):

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33,shuffle=False) # set the shuffle to false in case you want the data to be chronically orgainzed

    tensor_x_train = torch.Tensor(X_train).float().to(device)  # transform to torch tensor
    tensor_y_train = torch.Tensor(y_train).float().to(device)

    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)  # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # create your dataloader

    tensor_x_test = torch.Tensor(X_test).float().to(device)  # transform to torch tensor
    tensor_y_test = torch.Tensor(y_test).float().to(device)

    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
    test_dataloader = DataLoader(test_dataset)  # create your dataloader

    with open(os.path.join(main_path + '/'+data_type+'/segmented_data/' + file_version + '/train_dataloader_'+data_type + file_version + '.npy'), 'wb') as f:
        pickle.dump(train_dataloader, f, protocol=4)
    with open(os.path.join(main_path + '/'+data_type+'/segmented_data/' + file_version + '/test_dataloader_'+data_type + file_version + '.npy'), 'wb') as f:
        pickle.dump(test_dataloader, f, protocol=4)

    return train_dataloader, test_dataloader, y_test

def data_processing_all(data_pd,data_hc, labels_pd,labels_hc, batch_size):
    # split the data to train and test
    # split each group (hc/pd) separately, so the entire ("all") train & test set include participants from both group
    X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(data_pd, labels_pd, test_size=0.33,shuffle=False) # set the shuffle to false in case you want the data to be chronically orgainzed
    X_train_hc, X_test_hc, y_train_hc, y_test_hc = train_test_split(data_hc, labels_hc, test_size=0.33, shuffle=False)
    X_train=np.concatenate((X_train_pd,X_train_hc),axis=0)
    y_train=np.concatenate((y_train_pd,y_train_hc),axis=0)
    X_test=np.concatenate((X_test_pd,X_test_hc),axis=0)
    y_test=np.concatenate((y_test_pd,y_test_hc),axis=0)
    # shuffling windows- optional
    # train_randomlist = random.sample(range(X_train.shape[0]), X_train.shape[0])
    # test_randomlist = random.sample(range(X_test.shape[0]), X_test.shape[0])
    # X_train = X_train[train_randomlist, :, :, :]
    # y_train = y_train[train_randomlist, :]
    # X_test = X_test[test_randomlist, :, :, :]
    # y_test = y_test[test_randomlist, :]

    tensor_x_train = torch.Tensor(X_train).float().to(device)  # transform to torch tensor
    tensor_y_train = torch.Tensor(y_train).float().to(device)

    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)  # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # create your dataloader

    tensor_x_test = torch.Tensor(X_test).float().to(device)  # transform to torch tensor
    tensor_y_test = torch.Tensor(y_test).float().to(device)

    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
    test_dataloader = DataLoader(test_dataset)  # create your dataloader

    return train_dataloader, test_dataloader, y_test

def data_processing_CV(data_type,data_pd,data_hc, labels_pd,labels_hc,participants_pd,participants_hc,batch_size,main_path,file_version):
    # split the data to train and test
    # split each group (hc/pd) separately, so the entire ("all") train & test set include participants from both group
    pd_batch=np.array([[1,5],[5,9],[9,13],[13,16],[16,19] ])
    hc_batch = np.array([[1, 3], [3, 5], [5, 7], [7, 10], [10, 13]])
    for ind in range(5):
        torch.cuda.empty_cache()

        if (data_type=='all') or (data_type=='PD'):
            [IndexTest, _] = ismember(participants_pd, np.arange(pd_batch[ind][0], pd_batch[ind][1]))
            IndexTrain = IndexTest == 0
            X_train_pd = data_pd[IndexTrain[:, 1], :, :, :]
            X_test_pd = data_pd[IndexTest[:, 1], :, :, :]
            y_train_pd = labels_pd[IndexTrain[:, 1], :]
            y_test_pd = labels_pd[IndexTest[:, 1], :]
            # PD dataset
            tensor_x_train_pd = torch.Tensor(X_train_pd).float().to(device)  # transform to torch tensor
            tensor_y_train_pd = torch.Tensor(y_train_pd).float().to(device)
            train_dataset_pd = TensorDataset(tensor_x_train_pd, tensor_y_train_pd)  # create your datset
            train_dataloader_pd = DataLoader(train_dataset_pd, batch_size=batch_size,shuffle=True)  # create your dataloader
            tensor_x_test_pd = torch.Tensor(X_test_pd).float().to(device)  # transform to torch tensor
            tensor_y_test_pd = torch.Tensor(y_test_pd).float().to(device)
            test_dataset_pd = TensorDataset(tensor_x_test_pd, tensor_y_test_pd)  # create your datset
            test_dataloader_pd = DataLoader(test_dataset_pd)  # create your dataloader
            if data_type=='PD':
                with open(os.path.join(main_path + '/PD/segmented_data/' + file_version + '/train_dataloader_pd' + file_version + '_fold' + str(ind) + '.npy'), 'wb') as f:
                    pickle.dump(train_dataloader_pd, f, protocol=4)
                with open(os.path.join(main_path + '/PD/segmented_data/' + file_version + '/test_dataloader_pd'+ file_version + '_fold' + str(ind) + '.npy'), 'wb') as f:
                    pickle.dump(test_dataloader_pd, f, protocol=4)

        if (data_type == 'all') or (data_type =='HC'):
            [IndexTestHC, _] = ismember(participants_hc, np.arange(hc_batch[ind][0], hc_batch[ind][1]))
            IndexTrainHC = IndexTestHC == 0
            X_train_hc = data_hc[IndexTrainHC[:, 1], :, :, :]
            X_test_hc = data_hc[IndexTestHC[:, 1], :, :, :]
            y_train_hc = labels_hc[IndexTrainHC[:, 1], :]
            y_test_hc = labels_hc[IndexTestHC[:, 1], :]
            # HC dataset
            tensor_x_train_hc = torch.Tensor(X_train_hc).float().to(device)  # transform to torch tensor
            tensor_y_train_hc = torch.Tensor(y_train_hc).float().to(device)
            train_dataset_hc = TensorDataset(tensor_x_train_hc, tensor_y_train_hc)  # create your datset
            train_dataloader_hc = DataLoader(train_dataset_hc, batch_size=batch_size,shuffle=True)  # create your dataloader
            tensor_x_test_hc = torch.Tensor(X_test_hc).float().to(device)  # transform to torch tensor
            tensor_y_test_hc = torch.Tensor(y_test_hc).float().to(device)
            test_dataset_hc = TensorDataset(tensor_x_test_hc, tensor_y_test_hc)  # create your datset
            test_dataloader_hc = DataLoader(test_dataset_hc)  # create your
            if data_type=='HC':
                with open(os.path.join(main_path + '/HC/segmented_data/' + file_version + '/train_dataloader_hc' + file_version + '_fold' + str(ind) + '.npy'), 'wb') as f:
                    pickle.dump(train_dataloader_hc, f, protocol=4)
                with open(os.path.join(main_path + '/HC/segmented_data/' + file_version + '/test_dataloader_hc' + file_version + '_fold' + str(ind) + '.npy'), 'wb') as f:
                    pickle.dump(test_dataloader_hc,f, protocol=4)

        if data_type=='all':
            X_train = np.concatenate((X_train_pd, X_train_hc), axis=0)
            y_train = np.concatenate((y_train_pd, y_train_hc), axis=0)
            tensor_x_train = torch.Tensor(X_train).float().to(device)  # transform to torch tensor
            tensor_y_train = torch.Tensor(y_train).float().to(device)
            train_dataset = TensorDataset(tensor_x_train, tensor_y_train)  # create your datset
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # create your dataloader
            X_test = np.concatenate((X_test_pd, X_test_hc), axis=0)
            y_test = np.concatenate((y_test_pd, y_test_hc), axis=0)
            tensor_x_test = torch.Tensor(X_test).float().to(device)  # transform to torch tensor
            tensor_y_test = torch.Tensor(y_test).float().to(device)
            test_dataset = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  # create your dataloader
            with open(os.path.join(main_path + '/all/segmented_data/' + file_version + '/train_dataloader_all_' + file_version + '_fold' + str(ind) + '.npy'), 'wb') as f:
                pickle.dump(train_dataloader, f, protocol=4)
            with open(os.path.join(main_path + '/all/segmented_data/' + file_version + '/test_dataloader_all_' + file_version + '_fold' + str(ind) + '.npy'), 'wb') as f:
                pickle.dump(test_dataloader, f, protocol=4)
            with open(os.path.join(main_path + '/all/segmented_data/' + file_version + '/test_dataloader_pd_' + file_version + '_fold' + str(ind) + '.npy'), 'wb') as f:
                pickle.dump(test_dataloader_pd, f, protocol=4)
            with open(os.path.join(main_path + '/all/segmented_data/' + file_version + '/test_dataloader_hc_' + file_version + '_fold' + str(ind) + '.npy'),'wb') as f:
                pickle.dump(test_dataloader_hc, f, protocol=4)

def preprocessing(data,labels,fs,std_thresh,Resample,res_rate, comp_analysisis,Filtering):
    RMS=np.sqrt(np.sum(data**2,axis=1))
    RMS=pd.Series(RMS) #convert to pandas series for convenient moving std
    accel_std=RMS.rolling(fs).std() #calculate the std for each following #fs points of the data
    ind_potential_gait_std=accel_std.to_numpy()>std_thresh
    data_new = data[ind_potential_gait_std]
    labels_new = labels[ind_potential_gait_std]
    # data_reduced=data[ind_potential_gait_std==0]
    # labels_reduced=labels[ind_potential_gait_std==0]
    if Resample:
        data_new = signal.resample(data_new, len(data_new)*res_rate) #resample the signal to higher sampling rate for better resolution
        labels_new = np.round(signal.resample(labels_new, len(labels_new) * res_rate))
    if comp_analysisis:
        pca = PCA(n_components=2)
        acc_pca=pca.fit_transform(data_new)
        data_new=np.concatenate((acc_pca,np.zeros((len(acc_pca),1))),axis=1) #padding the 3-d axis in zeroes to fit the network architecture
    if Filtering:
        fc = 10 # Cut-off frequency of the filter
        w = fc / (fs / 2)  # Normalize the frequency
        b, a = signal.butter(2, w, 'low')
        data_new = signal.filtfilt(b, a, data_new.T)
    return data_new, labels_new, ind_potential_gait_std

##  Network Architecture
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[1, 16])
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 16])

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 16])
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 16])

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 16])
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 16])

        self.convTranspose1 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.conv2_5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 16])
        self.conv2_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 16])

        self.convTranspose2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.conv1_4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 16])
        self.conv1_4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 16])
        self.conv1_5 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[3, 1])
        self.conv1_6 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)

    # define forward function
    def forward(self, x):
        pad_x = nn.ReflectionPad2d((7, 8, 0, 0))(x)

        conv1_1 = F.relu(self.conv1_1(pad_x))
        conv1_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_1)
        conv1_2 = F.relu(self.conv1_2(conv1_1))

        conv2_1 = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])(conv1_2)

        conv2_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_1)
        conv2_2 = F.relu(self.conv2_1(conv2_1))
        conv2_2 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_2)
        conv2_3 = F.relu(self.conv2_2(conv2_2))

        conv3_1 = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])(conv2_3)

        conv3_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_1)
        conv3_2 = F.relu(self.conv3_1(conv3_1))
        conv3_2 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_2)
        conv3_3 = F.relu(self.conv3_2(conv3_2))
        conv3_3 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_3)
        conv3_4 = F.relu(self.conv3_2(conv3_3))

        conv2_4_1 = self.convTranspose1(conv3_4)
        conv2_4 = torch.cat((conv2_4_1, conv2_3), 1)
        conv2_4 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_4)
        conv2_5 = F.relu(self.conv2_5(conv2_4))
        conv2_5 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_5)
        conv2_6 = F.relu(self.conv2_6(conv2_5))

        conv1_3_1 = self.convTranspose2(conv2_6)
        conv1_3 = torch.cat((conv1_2, conv1_3_1), 1)
        conv1_3 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_3)
        conv1_4 = F.relu(self.conv1_4(conv1_3))
        conv1_4 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_4)
        conv1_4 = F.relu(self.conv1_4_2(conv1_4))
        conv1_5 = F.relu(self.conv1_5(conv1_4))
        conv1_6 = torch.sigmoid(self.conv1_6(conv1_5))

        out = torch.reshape(conv1_6, (-1, 512))

        return out

# class NetworkBatchNorm(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[1, 16])
#         self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 16])
#
#         self.batchNormconv1 = nn.BatchNorm2d(64)
#
#         self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 16])
#         self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 16])
#         self.batchNormconv2 = nn.BatchNorm2d(128)
#
#         self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 16])
#         self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 16])
#         self.batchNormconv3 = nn.BatchNorm2d(256)
#
#         self.convTranspose1 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
#         self.conv2_5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 16])
#         self.conv2_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 16])
#         self.batchNormconv4 = nn.BatchNorm2d(128)
#
#         self.convTranspose2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
#         self.conv1_4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 16])
#         self.conv1_4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 16])
#         self.conv1_5 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[3, 1])
#         self.conv1_6 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
#
#
#     # define forward function
#     def forward(self, x):
#         pad_x = nn.ReflectionPad2d((7, 8, 0, 0))(x)
#
#         conv1_1 = F.relu(self.conv1_1(pad_x))
#         conv1_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_1)
#         conv1_1 = self.batchNormconv1(conv1_1)
#         conv1_2 = F.relu(self.conv1_2(conv1_1))
#
#         conv2_1 = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])(conv1_2)
#
#         conv2_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_1)
#         conv2_2 = F.relu(self.conv2_1(conv2_1))
#         conv2_2 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_2)
#         conv2_2 = self.batchNormconv2(conv2_2)
#         conv2_3 = F.relu(self.conv2_2(conv2_2))
#
#         conv3_1 = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])(conv2_3)
#
#         conv3_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_1)
#         conv3_2 = F.relu(self.conv3_1(conv3_1))
#         conv3_2 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_2)
#         conv3_2 = self.batchNormconv3(conv3_2)
#         conv3_3 = F.relu(self.conv3_2(conv3_2))
#         conv3_3 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_3)
#         conv3_4 = F.relu(self.conv3_2(conv3_3))
#
#         conv2_4_1 = self.convTranspose1(conv3_4)
#         conv2_4 = torch.cat((conv2_4_1, conv2_3), 1)
#         conv2_4 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_4)
#         conv2_4 = self.batchNormconv4(conv2_4)
#         conv2_5 = F.relu(self.conv2_5(conv2_4))
#         conv2_5 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_5)
#         conv2_6 = F.relu(self.conv2_6(conv2_5))
#
#         conv1_3_1 = self.convTranspose2(conv2_6)
#         conv1_3 = torch.cat((conv1_2, conv1_3_1), 1)
#         conv1_3 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_3)
#         conv1_3 = self.batchNormconv4(conv1_3)
#         conv1_4 = F.relu(self.conv1_4(conv1_3))
#         conv1_4 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_4)
#         conv1_4 = F.relu(self.conv1_4_2(conv1_4))
#         conv1_5 = F.relu(self.conv1_5(conv1_4))
#
#         conv1_6 = torch.sigmoid(self.conv1_6(conv1_5))
#
#         out = torch.reshape(conv1_6, (-1, 512))
#
#         return out
class NetworkBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[1, 16])
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 16])

        self.batchNormconv1 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 16])
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 16])

        self.batchNormconv2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 16])
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 16])

        self.batchNormconv3 = nn.BatchNorm2d(256)

        self.convTranspose1 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.conv2_5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 16])
        self.conv2_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 16])

        self.convTranspose2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.conv1_4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 16])
        self.conv1_4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 16])
        self.conv1_5 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[3, 1])
        self.conv1_6 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)

    # define forward function
    def forward(self, x):
        pad_x = nn.ReflectionPad2d((7, 8, 0, 0))(x)

        conv1_1 = F.relu(self.conv1_1(pad_x))
        conv1_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_1)
        conv1_1 = self.batchNormconv1(conv1_1)
        conv1_2 = F.relu(self.conv1_2(conv1_1))

        conv2_1 = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])(conv1_2)

        conv2_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_1)
        conv2_2 = F.relu(self.conv2_1(conv2_1))
        conv2_2 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_2)
        conv2_2 = self.batchNormconv2(conv2_2)
        conv2_3 = F.relu(self.conv2_2(conv2_2))

        conv3_1 = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])(conv2_3)

        conv3_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_1)
        conv3_2 = F.relu(self.conv3_1(conv3_1))
        conv3_2 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_2)
        conv3_2 = self.batchNormconv3(conv3_2)
        conv3_3 = F.relu(self.conv3_2(conv3_2))
        conv3_3 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_3)
        conv3_4 = F.relu(self.conv3_2(conv3_3))

        conv2_4_1 = self.convTranspose1(conv3_4)
        conv2_4 = torch.cat((conv2_4_1, conv2_3), 1)
        conv2_4 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_4)
        conv2_5 = F.relu(self.conv2_5(conv2_4))
        conv2_5 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_5)
        conv2_6 = F.relu(self.conv2_6(conv2_5))

        conv1_3_1 = self.convTranspose2(conv2_6)
        conv1_3 = torch.cat((conv1_2, conv1_3_1), 1)
        conv1_3 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_3)
        conv1_4 = F.relu(self.conv1_4(conv1_3))
        conv1_4 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_4)
        conv1_4 = F.relu(self.conv1_4_2(conv1_4))
        conv1_5 = F.relu(self.conv1_5(conv1_4))
        conv1_6 = torch.sigmoid(self.conv1_6(conv1_5))

        out = torch.reshape(conv1_6, (-1, 512))

        return out

class NetworkBatchNormDrop(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[1, 16])
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 16])

        self.batchNormconv1 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 16])
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 16])
        self.batchNormconv2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 16])
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 16])
        self.batchNormconv3 = nn.BatchNorm2d(256)

        self.convTranspose1 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.conv2_5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 16])
        self.conv2_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 16])
        self.batchNormconv4 = nn.BatchNorm2d(128)

        self.convTranspose2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.conv1_4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 16])
        self.conv1_4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 16])
        self.conv1_5 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[3, 1])
        self.conv1_6 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)

        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)



    # define forward function
    def forward(self, x):
        pad_x = nn.ReflectionPad2d((7, 8, 0, 0))(x)

        conv1_1 = F.relu(self.conv1_1(pad_x))
        conv1_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_1)
        conv1_1 = self.batchNormconv1(conv1_1)
        conv1_2 = F.relu(self.conv1_2(conv1_1))

        conv2_1 = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])(conv1_2)
        conv2_1= self.drop1(conv2_1)
        conv2_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_1)
        conv2_2 = F.relu(self.conv2_1(conv2_1))
        conv2_2 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_2)
        conv2_2 = self.batchNormconv2(conv2_2)
        conv2_3 = F.relu(self.conv2_2(conv2_2))

        conv3_1 = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])(conv2_3)
        conv3_1 = self.drop1(conv3_1)
        conv3_1 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_1)
        conv3_2 = F.relu(self.conv3_1(conv3_1))
        conv3_2 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_2)
        conv3_3 = F.relu(self.conv3_2(conv3_2))
        conv3_3 = nn.ReflectionPad2d((7, 8, 0, 0))(conv3_3)
        conv3_3 = self.batchNormconv3(conv3_3)
        conv3_4 = F.relu(self.conv3_2(conv3_3))

        conv2_4_1 = self.convTranspose1(conv3_4)
        conv2_4_1 = self.drop1(conv2_4_1)
        conv2_4 = torch.cat((conv2_4_1, conv2_3), 1)
        conv2_4 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_4)
        conv2_5 = F.relu(self.conv2_5(conv2_4))
        conv2_5 = nn.ReflectionPad2d((7, 8, 0, 0))(conv2_5)
        conv2_5 = self.batchNormconv4(conv2_5)
        conv2_6 = F.relu(self.conv2_6(conv2_5))

        conv1_3_1 = self.convTranspose2(conv2_6)
        conv1_3_1 = self.drop1(conv1_3_1)
        conv1_3 = torch.cat((conv1_2, conv1_3_1), 1)
        conv1_3 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_3)
        conv1_3 = self.batchNormconv4(conv1_3)
        conv1_4 = F.relu(self.conv1_4(conv1_3))
        conv1_4 = nn.ReflectionPad2d((7, 8, 0, 0))(conv1_4)
        conv1_4 = F.relu(self.conv1_4_2(conv1_4))
        conv1_5 = F.relu(self.conv1_5(conv1_4))
        conv1_5=self.drop2(conv1_5)
        conv1_6 = torch.sigmoid(self.conv1_6(conv1_5))

        out = torch.reshape(conv1_6, (-1, 512))

        return out

def my_loss(y_,output_map):
  t0 = y_*torch.log(torch.clamp(output_map,1e-10,1.0))
  t1 = (1-y_)*torch.log(torch.clamp(1-output_map,1e-10,1.0))
  return -torch.mean( t0 + t1)
  #return -torch.mean( y_*torch.log(torch.clamp(output_map,1e-10,1.0)) +(1-y_)*torch.log(torch.clamp(1-output_map,1e-10,1.0)))

def check_acc(model, loader):
    correct = 0
    total = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    all_outputs=torch.empty((0, 512)).to(device)
    with torch.no_grad():
        for data in loader:
            features, labels = data
            # if len(outputs) == 0:
            outputs = model(features)
            all_outputs=torch.cat((all_outputs,outputs),dim=0)
            predicted = torch.round(outputs)
            total += (labels.size(0) * labels.size(1))
            correct += torch.sum(predicted == labels)
            TP += torch.sum(((predicted == labels) & (labels == 1)))
            TN += torch.sum(((predicted == labels) & (labels == 0)))
            FP += torch.sum(((predicted != labels) & (labels == 0)))
            FN += torch.sum(((predicted != labels) & (labels == 1)))
    acc = (100 * correct / total)
    precision = 100 * ((1+TP) / (1 + TP + FP))
    sensitivity = 100 * ((1+TP) / (1 + TP + FN))
    specificity = 100 * ((1+TN) / (1 + TN + FP))

    # if loader == 'train_dataloader':
    #     print('Accuracy on the training set is : %.3f %%' % acc)
    # elif loader == 'test_dataloader':
    #     print('Accuracy on the validation set is: %.3f %%' % acc)

    return all_outputs, [acc, precision, sensitivity, specificity]

# training
def train_model_loop_all(model, optimizer, num_steps, train_dataloader, test_dataloader_pd,test_dataloader_hc,balance):
    all_train_loss = []
    # all_test_loss = []
    train_acc = []
    validation_acc = []

    for epoch in range(num_steps):
        model.train()
        # train_preds = torch.empty((0,512))
        for batch_idx, (data, labels) in enumerate(train_dataloader):
            preds = model(data)
            if balance=="Yes":
                # train_preds=torch.cat((train_preds,preds),dim=0)
                imbalance_factor = torch.floor(len(labels.reshape(-1)) / torch.sum(labels.reshape(-1)))
                class_weights = torch.ones_like(labels) / imbalance_factor + (1.0 - 1.0 / imbalance_factor) * labels
                loss = torch.nn.functional.binary_cross_entropy(preds, labels, class_weights)

            else:
                loss = my_loss(labels, preds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bach_loss = loss.item()

            if batch_idx % 100 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch, batch_idx, bach_loss))

        all_train_loss.append(float(bach_loss))
        model.eval()
        print("epoch %d,  train loss : %.3f,"  %(epoch, bach_loss))
        # train_acc.append(check_acc(model, train_dataloader))
        # validation_acc.append(check_acc(model, test_dataloader))
    _,train_acc=check_acc(model, train_dataloader)
    validation_outputs_pd,validation_acc_pd=check_acc(model, test_dataloader_pd)
    validation_outputs_hc,validation_acc_hc = check_acc(model, test_dataloader_hc)
    return model, all_train_loss,train_acc, validation_acc_pd,validation_acc_hc,validation_outputs_pd,validation_outputs_hc

def train_model_loop(model, optimizer, num_steps, train_dataloader, test_dataloader,balance):
    all_train_loss = []
    all_test_loss = []
    all_train_acc = []
    all_validation_acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_steps):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data=data.to(device)
            labels=labels.to(device)
            preds = model(data)
            if balance=="Yes":
                imbalance_factor = torch.floor(len(labels.reshape(-1)) / torch.sum(labels.reshape(-1)))
                # imbalance_factor=imbalance_factor-0.2*imbalance_factor #when balancing the model tend to bias against the positive labels (gait, i.e, high recall but low precision). Then, penalizing the imbalance factor for more stable results
                class_weights = torch.ones_like(labels) / imbalance_factor + (1.0 - 1.0 / imbalance_factor) * labels
                loss = torch.nn.functional.binary_cross_entropy(preds, labels, class_weights)

            else:
                loss = my_loss(labels, preds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bach_loss = loss.item()

            if batch_idx % 100 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch, batch_idx, bach_loss))

        all_train_loss.append(float(bach_loss))
        model.eval()
        with torch.no_grad():
            total = 0
            total_loss = 0
            for data, labels in test_dataloader:
                data=data.to(device)
                labels=labels.to(device)
                test_preds = model(data)
                test_loss = my_loss(labels, test_preds)
                total += labels.size(0)
                total_loss+=test_loss.item()
            all_test_loss.append(total_loss/total)
            print("epoch %d,  train loss : %.3f,"  %(epoch, bach_loss))
            print("epoch %d,  test loss : %.3f," % (epoch, (total_loss/total)))
            _, train_acc = check_acc(model, train_dataloader)
            _, validation_acc = check_acc(model, test_dataloader)
            all_train_acc.append(train_acc)
            all_validation_acc.append(validation_acc)
    validation_outputs, _ = check_acc(model, test_dataloader)
    return model, all_train_loss,all_test_loss,all_train_acc,all_validation_acc,validation_outputs

# model performance
def post_processing(input_path,data_type,file_version,file2load,win,fs,preds_thr):
    labels = open(os.path.join(input_path+'/'+data_type+'/segmented_data/'+file2load+'/'+data_type.lower()+'_labels'+file2load+'.npy'), "rb")
    labels = pickle.load(labels)
    labels=labels.reshape(-1)
    preds = open(os.path.join(input_path+'/'+data_type+'/model/'+file_version+'/all_preds'+file_version+'.npy'), "rb")
    preds = pickle.load(preds)

    # round to 0/1 to get classification from the prediction logits
    new_preds=np.zeros_like(preds)
    new_preds[preds.numpy()>preds_thr]=1
    # found the start and stop point of each predicted gait bout
    GaitBoutsStartStop = np.concatenate((np.asarray(
        np.where(np.diff(np.concatenate(([0], new_preds, [0]))) == 1)), np.asarray(
        np.where(np.diff(np.concatenate(([0], new_preds, [0]))) == -1)) - 1), axis=0)

    rngs = GaitBoutsStartStop[1, :] - GaitBoutsStartStop[0, :]  # bouts durations
    for bout in range(1, GaitBoutsStartStop.shape[1]):
        # merge bouts with interval less than 1 sec
        if GaitBoutsStartStop[0, bout] - GaitBoutsStartStop[1, bout - 1] < fs:
            new_preds[GaitBoutsStartStop[1, bout - 1]:GaitBoutsStartStop[0, bout]] = 1

    # update the start/end points after merging
    GaitBoutsStartStop = np.concatenate((np.asarray(
        np.where(np.diff(np.concatenate(([0], new_preds, [0]))) == 1)), np.asarray(
        np.where(np.diff(np.concatenate(([0], new_preds, [0]))) == -1)) - 1), axis=0)
    rngs_new = GaitBoutsStartStop[1, :] - GaitBoutsStartStop[0, :]
    GaitBoutsStartStop = GaitBoutsStartStop[:, rngs_new > win * fs]  # remove bouts with duration less than #win sec
    preds_final = np.zeros_like(preds)

    for ind in range(GaitBoutsStartStop.shape[1]):
        preds_final[GaitBoutsStartStop[0, ind]:GaitBoutsStartStop[1, ind]] = 1


    TP=np.sum((preds_final == labels) & (labels == 1))
    TN=np.sum((preds_final==labels) & (labels==0))
    FP=np.sum((preds_final != labels) & (labels == 0))
    FN = np.sum((preds_final != labels) & (labels == 1))
    # acc = 100 * (TP+TN)/(TP+TN+FP+FN)
    precision = 100 * ((1+TP) / (1 + TP + FP))
    sensitivity = 100 * ((1+TP) / (1 + TP + FN))
    specificity = 100 * ((1+TN) / (1 + TN + FP))

    with open(os.path.join(input_path+'/'+data_type+'/model/'+file_version+'/results_Postprocessing'+file_version+'.npy'), 'wb') as f:
        pickle.dump([precision,sensitivity,specificity ], f, protocol=4)
    with open(os.path.join(input_path+'/'+data_type+'/model/'+file_version+'/final_outputs'+file_version+'.npy'), 'wb') as f:
        pickle.dump(preds_final, f, protocol=4)

def precision_recall(input_path,data_type, file_version,file2load,CV):
    if data_type=='physionet':
        test_preds = open(
            os.path.join((input_path + '/model/all_preds' + file_version + '.npy')),
            "rb")
        test_preds = pickle.load(test_preds)
        test_labels = open(os.path.join(input_path + '/segmented_data/y_test_physio1.npy'), "rb")
        test_labels = pickle.load(test_labels)
        test_preds = test_preds.reshape(-1).detach()
        test_labels = test_labels.reshape(-1)
        precision, recall, thresholds = precision_recall_curve(test_labels, test_preds)

        fig, ax = plt.subplots()
        ax.step(recall, precision, color='r', alpha=0.99, where='post')
        ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        viz = Visdom()
        imgdata = io.StringIO()
        fig.savefig(imgdata, format='svg')

        svg_str = imgdata.getvalue()
        # Scale the figure
        svg_str = re.sub('width=".*pt"', 'width="100%"', svg_str)
        svg_str = re.sub('height=".*pt"', 'height="100%"', svg_str)
        viz.svg(svg_str)
        fig.savefig(os.path.join(input_path +  '/model/precision_recall' + file_version + '.png'))

        auc_score = auc(recall, precision)

        # calculate roc curve
        fpr, tpr, _ = roc_curve(test_labels, test_preds)

        fig, ax = plt.subplots()
        ax.step(fpr, tpr, color='r', alpha=0.99, where='post')
        ax.fill_between(fpr, tpr, alpha=0.2, color='b', step='post')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        viz = Visdom()
        imgdata = io.StringIO()
        fig.savefig(imgdata, format='svg')

        svg_str = imgdata.getvalue()
        # Scale the figure
        svg_str = re.sub('width=".*pt"', 'width="100%"', svg_str)
        svg_str = re.sub('height=".*pt"', 'height="100%"', svg_str)
        viz.svg(svg_str)
        fig.savefig(
            os.path.join(input_path +'/model/auc_roc' + file_version + '.png'))
        roc_auc = roc_auc_score(test_labels, test_preds)
        auc_scores = {"roc_auc": roc_auc, "precision_recall_auc": auc_score}
        with open(os.path.join(
                input_path +  '/model/precision_recall_auc' + file_version + '.npy'),
                  'wb') as f:
            pickle.dump(auc_scores, f, protocol=4)
    elif CV:
        preds = open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/all_preds' + file_version + '.npy'), "rb")
        preds = pickle.load(preds)

        labels = open(os.path.join(input_path + '/' + data_type + '/segmented_data/' + file2load + '/' +data_type.lower()+'_labels'+ file2load + '.npy'), "rb")
        labels = pickle.load(labels)
        labels=labels.reshape(-1)

        #Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(labels, preds)

        PR_auc = auc(recall, precision) #area under the PR curve

        # calculate roc curve
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = roc_auc_score(labels, preds)

        auc_scores = {"roc_auc": roc_auc, "precision_recall_auc": PR_auc}

        with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/auc_scores' + file_version + '.npy'),'wb') as f:
            pickle.dump(auc_scores, f, protocol=4)

        # Plot the curves
        fig = plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            label=data_type+": deep convolutional network (area = {0:0.2f})".format(roc_auc),
            color="deeppink",
            linestyle="-",
            linewidth=4,
        )

        plt.plot([1, 0], [1, 0], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/ROC_fig' + file_version + '.png'))
        plt.show()

        fig2 = plt.figure()
        lw = 2
        plt.plot(
            recall,
            precision,
            label= data_type+": deep convolutional network (area = {0:0.2f})".format(PR_auc),
            color="deeppink",
            linestyle="-",
            linewidth=4
        )


        plt.plot([1, 0], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/PR_fig' + file_version + '.png'))
        plt.show()

    else:
        test_preds = open(os.path.join(input_path +'/'+ data_type + '/model/' + file_version + '/all_preds' + file_version + '.npy'),"rb")
        test_preds = pickle.load(test_preds)
        test_labels = open(os.path.join(input_path + '/all/segmented_data/y_test_all' + file_version + '.npy'), "rb")
        test_labels = pickle.load(test_labels)
        test_preds = test_preds.reshape(-1).detach()
        test_labels = test_labels.reshape(-1)
        precision, recall, thresholds = precision_recall_curve(test_labels, test_preds)
        locs=np.around(np.linspace(0,(len(recall)-1),10000)).astype(int)

        fig, ax = plt.subplots()
        ax.step(recall[locs], precision[locs], color='r', alpha=0.99, where='post')
        ax.fill_between(recall[locs], precision[locs], alpha=0.2, color='b', step='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        viz = Visdom()
        imgdata = io.StringIO()
        fig.savefig(imgdata, format='svg')

        svg_str = imgdata.getvalue()
        # Scale the figure
        svg_str = re.sub('width=".*pt"', 'width="100%"', svg_str)
        svg_str = re.sub('height=".*pt"', 'height="100%"', svg_str)
        viz.svg(svg_str)
        fig.savefig(os.path.join(
            input_path +'/'+ data_type + '/model/' + file_version + '/precision_recall' + file_version + '.png'))

        auc_score = auc(recall, precision)

        # calculate roc curve
        fpr, tpr, _ = roc_curve(test_labels, test_preds)

        fig, ax = plt.subplots()
        ax.step(fpr, tpr, color='r', alpha=0.99, where='post')
        ax.fill_between(fpr, tpr, alpha=0.2, color='b', step='post')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        viz = Visdom()
        imgdata = io.StringIO()
        fig.savefig(imgdata, format='svg')

        svg_str = imgdata.getvalue()
        # Scale the figure
        svg_str = re.sub('width=".*pt"', 'width="100%"', svg_str)
        svg_str = re.sub('height=".*pt"', 'height="100%"', svg_str)
        viz.svg(svg_str)
        fig.savefig(
            os.path.join(input_path +'/' +data_type + '/model/' + file_version + '/auc_roc' + file_version + '.png'))
        roc_auc = roc_auc_score(test_labels, test_preds)
        auc_scores = {"roc_auc": roc_auc, "precision_recall_auc": auc_score}
        with open(os.path.join(
                input_path +'/'+ data_type + '/model/' + file_version + '/precision_recall_auc' + file_version + '.npy'),
                  'wb') as f:
            pickle.dump(auc_scores, f, protocol=4)

def daily_activity_corr(input_path,data_type,file_version,file2load,preds_thr,subject_level):
    preds = open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/all_preds' + file_version + '.npy'),"rb")
    preds = pickle.load(preds)
    new_preds=np.zeros_like(preds)
    new_preds[preds.numpy()>preds_thr]=1 # you can change the threshold of the round according to the ROC /PR performance

    labels = open(os.path.join(input_path + '/' + data_type + '/segmented_data/' + file2load + '/' + data_type.lower() + '_labels' + file2load + '.npy'),"rb")
    labels = pickle.load(labels)
    labels = labels.reshape(-1)

    dayStartStop = sio.loadmat(os.path.join(input_path + '/' + data_type + '/dayStartStop'+data_type+'.mat')) #matrix including the start and the end point of each day of the valid days (8hr+ of daily acitivity) in the data
    dayStartStop = dayStartStop['dayStartStop']
    
    if subject_level=="True":
        activity_mat = np.zeros((2, len(np.unique(dayStartStop[2, :]))))
        count = 0
        for subject in np.unique(dayStartStop[2, :]):
            subMat = dayStartStop[:2, dayStartStop[2, :] == subject]
            day_activity = np.zeros((2, subMat.shape[1]))
            for day in range(subMat.shape[1]):
                day_activity[0, day] = sum(labels[subMat[0, day]:subMat[1, day]])  # real activity
                day_activity[1, day] = sum(new_preds[subMat[0, day]:subMat[1, day]])  # predicted activity
            activity_mat[0, count] = np.median(day_activity[0, :])
            activity_mat[1, count] = np.median(day_activity[1, :])
            count += 1
    else:
        activity_mat = np.zeros((2, dayStartStop.shape[1]))
        for day in range(dayStartStop.shape[1]):
            activity_mat[0, day] = sum(labels[dayStartStop[0, day]:dayStartStop[1, day]])  # real activity
            activity_mat[1, day] = sum(new_preds[dayStartStop[0, day]:dayStartStop[1, day]])  # predicted activity

    activity_mat=activity_mat/(25*60)
    corr, _ = pearsonr(activity_mat[0, :], activity_mat[1, :])
    plt.figure()
    plt.scatter(activity_mat[0, :], activity_mat[1, :], label="Pearson's correlation coefficient = {0:0.2f}".format(corr),)
    # plt.title('Association between gold standard and the algorithm' "'" 's output: '+ data_type)
    plt.title('Association between gold standard and the algorithm' "'" 's output:'+ data_type)
    plt.xlabel('Real activity (minutes)')
    plt.ylabel('Predicted activity (minutes)')
    plt.legend(loc="lower right")
    plt.grid(False)
    plt.savefig(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/correlation_fig' + file_version + '.jpeg'),dpi=300)
    plt.show()

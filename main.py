# Gait segmentation
# This code is aims to detect gait from raw accelerometry data using semantic segmantation model (Unet like model)
#This model based on the paper: Zou Q, Wang Y, Zhao Y, Wang Q and Li Q, Deep learning-based gait recogntion using smartphones in the wild, IEEE Transactions on Information Forensics and Security, vol. 15, no. 1, pp. 3197-3212, 2020

import gait_segmentation
import numpy as np
import sys
import pickle
import torch
import torch.optim as optim
import os


def process():
    input_path = sys.argv[2]
    batch_size = int(sys.argv[3])
    file_version = sys.argv[4]
    data_type = sys.argv[5]
    CV=sys.argv[6] # cross validation- "Yes""/"No"
    fs=int(sys.argv[7]) #sampling rate
    preprocess=sys.argv[8]
    # std_thresh=float(sys.argv[8])
    # Resample=sys.argv[9]
    # res_rate=int(sys.argv[10])
    # comp_analysis=sys.argv[11]
    # Filtering=sys.argv[12]

    main_path = input_path
    if preprocess=="Yes":
        PD_path = os.path.join(main_path + '/Preprocessed/PD/' + str(fs))
        HC_path = os.path.join(main_path + '/Preprocessed/HC/' + str(fs))
    else:
        PD_path = os.path.join(main_path + '/PD/')
        HC_path = os.path.join(main_path + '/HC/')


    data_path_pd = PD_path + '/Data'
    label_path_pd = PD_path + '/labels'

    data_path_hc = HC_path + '/Data'
    label_path_hc = HC_path + '/labels'


    if data_type=="PD":
        # read the data and labels- PD
        data_pd = gait_segmentation.read_data(data_path_pd, fs)
        labels_pd, participants_pd = gait_segmentation.read_labels(label_path_pd, fs)
        # data_pd,labels_pd,participants_pd,ind_potential_gait_pd=gait_segmentation.read_data_labels(data_path_pd,label_path_pd,data_type,fs,std_thresh, Resample, res_rate, comp_analysis, Filtering)
        if os.path.isdir(os.path.join(PD_path + '/segmented_data/'+file_version))==0:
            os.mkdir(os.path.join(PD_path + '/segmented_data/'+file_version)) #open new directory if not existed
        with open(os.path.join(main_path + '/PD/segmented_data/' + file_version + '/pd_data' + file_version + '.npy'),'wb') as f:
            pickle.dump(data_pd, f, protocol=4)
        with open(os.path.join(main_path + '/PD/segmented_data/' + file_version + '/pd_labels' + file_version + '.npy'),'wb') as f:
            pickle.dump(labels_pd, f, protocol=4)
        with open(os.path.join(main_path + '/PD/segmented_data/' + file_version + '/pd_participants' + file_version + '.npy'),'wb') as f:
            pickle.dump(participants_pd, f, protocol=4)
        # with open(os.path.join(main_path + '/PD/segmented_data/' + file_version + '/pd_gait_ind' + file_version + '.npy'),'wb') as f:
        #     pickle.dump(ind_potential_gait_pd, f, protocol=4)

        if CV=="Yes":
            gait_segmentation.data_processing_CV(data_type,data_pd,[], labels_pd,[],participants_pd,[],batch_size,main_path,file_version)
        else:
            gait_segmentation.data_processing(main_path,file_version,data_type,data_pd, labels_pd, batch_size)

    elif data_type=="HC":
        if os.path.isdir(os.path.join(main_path + '/HC/segmented_data/'+file_version))==0:
            os.mkdir(os.path.join(main_path + '/HC/segmented_data/'+file_version))
        # data_hc, labels_hc, participants_hc, ind_potential_gait_hc = gait_segmentation.read_data_labels(data_path_hc,label_path_hc,data_type, fs,std_thresh,Resample,res_rate,comp_analysis,Filtering)
        # read the data and labels-HC
        data_hc = gait_segmentation.read_data(data_path_hc, fs)
        labels_hc, participants_hc = gait_segmentation.read_labels(label_path_hc, fs)
        with open(os.path.join(main_path + '/HC/segmented_data/' + file_version + '/hc_labels' + file_version + '.npy'),'wb') as f:
            pickle.dump(labels_hc, f, protocol=4)
        with open(os.path.join(main_path + '/HC/segmented_data/' + file_version + '/hc_data' + file_version + '.npy'),'wb') as f:
            pickle.dump(data_hc, f, protocol=4)
        with open(os.path.join(main_path + '/hc/segmented_data/' + file_version + '/hc_participants' + file_version + '.npy'),'wb') as f:
            pickle.dump(participants_hc, f, protocol=4)
        # with open(os.path.join(main_path + '/hc/segmented_data/' + file_version + '/hc_gait_ind' + file_version + '.npy'),'wb') as f:
        #     pickle.dump(ind_potential_gait_hc, f, protocol=4)
        if CV=="Yes":
            gait_segmentation.data_processing_CV(data_type,[],data_hc, [],labels_hc,[],participants_hc,batch_size,main_path,file_version)
        else:
            gait_segmentation.data_processing(main_path,file_version,data_type,data_hc, labels_hc, batch_size)

    elif data_type=="all":
        # read the data and labels- PD
        data_pd = gait_segmentation.read_data(data_path_pd, fs)
        labels_pd, participants_pd = gait_segmentation.read_labels(label_path_pd, fs)

        # read the data and labels-HC
        data_hc = gait_segmentation.read_data(data_path_hc, fs)
        labels_hc, participants_hc = gait_segmentation.read_labels(label_path_hc,  fs)

        # concatenate pd and hc data&labels
        all_data = np.concatenate((data_pd, data_hc), axis=0)
        all_labels = np.concatenate((labels_pd, labels_hc), axis=0)
        all_participants = np.concatenate((participants_pd, participants_hc), axis=0)
        if os.path.isdir(os.path.join(main_path + '/all/segmented_data/' + file_version)) == 0:
            os.mkdir(os.path.join(main_path + '/all/segmented_data/' + file_version))
        with open(os.path.join(main_path + '/all/segmented_data/' + file_version + '/pd_labels' + file_version + '.npy'),'wb') as f:
            pickle.dump(labels_pd, f, protocol=4)
        with open(os.path.join(main_path + '/all/segmented_data/' + file_version + '/hc_labels' + file_version + '.npy'),'wb') as f:
            pickle.dump(labels_hc, f, protocol=4)
        with open(os.path.join(main_path + '/all/segmented_data/' + file_version + '/all_labels' + file_version + '.npy'),'wb') as f:
            pickle.dump(all_labels, f, protocol=4)
        with open(os.path.join(main_path + '/all/segmented_data/' + file_version + '/all_data' + file_version + '.npy'),'wb') as f:
            pickle.dump(all_data, f, protocol=4)
        with open(os.path.join(main_path + '/all/segmented_data/' + file_version + '/all_participants' + file_version + '.npy'),'wb') as f:
            pickle.dump(all_participants, f, protocol=4)
        if CV=="Yes":
            gait_segmentation.data_processing_CV(data_type, data_pd, data_hc, labels_pd, labels_hc, participants_pd, participants_hc,batch_size, main_path, file_version)
        else:
            gait_segmentation.data_processing_all(data_pd,data_hc,labels_pd,labels_hc,batch_size)

def process_physionet():
    input_path = sys.argv[2]
    down_sample=sys.argv[3]
    gait_segmentation.read_physionet(input_path, down_sample)

    data = open(os.path.join(input_path + '/data_physionet_'+down_sample+'.npy'), "rb")
    data = pickle.load(data)
    labels = open(os.path.join(input_path + '/labels_physionet_'+down_sample+'.npy'), "rb")
    labels = pickle.load(labels)
    participants = open(os.path.join(input_path + '/participants_physionet_'+down_sample+'.npy'), "rb")
    participants = pickle.load(participants)

    batch_size = int(sys.argv[4])
    train_dataloader_physio, test_dataloader_physio, y_test_physio = gait_segmentation.data_processing(data, labels,
                                                                                                       batch_size)
    with open(os.path.join(input_path + '/segmented_data/physio_train_loader'+down_sample+'.npy'), 'wb') as f:
        pickle.dump(train_dataloader_physio, f, protocol=4)
    with open(os.path.join(input_path + '/segmented_data/physio_test_loader'+down_sample+'.npy'), 'wb') as f:
        pickle.dump(test_dataloader_physio, f, protocol=4)
    with open(os.path.join(input_path + '/segmented_data/y_test_physio'+down_sample+'.npy'), 'wb') as f:
        pickle.dump(y_test_physio, f, protocol=4)

def load_data(input_path, data_type, file_version):
    train_loader = open(os.path.join(input_path +'/'+ data_type + '/segmented_data/'+file_version+ '/train_dataloader_' +data_type+ file_version + '.npy'), "rb")
    train_loader = pickle.load(train_loader)
    test_loader = open(os.path.join(input_path +'/'+ data_type + '/segmented_data/'+file_version+'/test_dataloader_' + data_type + file_version + '.npy'),"rb")
    test_loader = pickle.load(test_loader)

    # elif data_type == "physionet":
    #     train_loader = open(os.path.join(input_path + '/segmented_data/physio_train_loaderdown_sample.npy'), "rb")
    #     train_loader = pickle.load(train_loader)
    #     test_loader = open(os.path.join(input_path + '/segmented_data/physio_test_loaderdown_sample.npy'), "rb")
    #     test_loader = pickle.load(test_loader)
    return train_loader, test_loader

def load_dataCV(input_path, data_type, file_version,fold):
    train_loader = open(os.path.join(input_path +'/'+ data_type + '/segmented_data/'+file_version+'/train_dataloader_'+data_type.lower() + file_version +'_fold'+fold+ '.npy'), "rb")
    train_loader = pickle.load(train_loader)
    test_loader= open(os.path.join(input_path +'/'+ data_type + '/segmented_data/'+file_version+'/test_dataloader_'+data_type.lower() + file_version +'_fold'+fold+ '.npy'), "rb")
    test_loader= pickle.load(test_loader)
    return  train_loader,test_loader

def load_dataCVall(input_path, data_type, file_version,fold):
    train_loader = open(os.path.join(input_path +'/'+ data_type + '/segmented_data/'+file_version+'/train_dataloader_all' + file_version +'_fold'+fold+ '.npy'), "rb")
    train_loader = pickle.load(train_loader)
    test_loader_pd = open(os.path.join(input_path +'/'+ data_type + '/segmented_data/'+file_version+'/test_dataloader_pd_' + file_version +'_fold'+fold+ '.npy'), "rb")
    test_loader_pd = pickle.load(test_loader_pd)
    test_loader_hc = open(os.path.join(input_path + '/' + data_type + '/segmented_data/' + file_version + '/test_dataloader_hc_' + file_version + '_fold' + fold + '.npy'),"rb")
    test_loader_hc = pickle.load(test_loader_hc)
    return  train_loader,test_loader_pd,test_loader_hc

def main():
    mode = sys.argv[1]
    if mode == "process":
        process()
    elif mode == "process_physionet":
        process_physionet()
    elif mode == "models":
        data_type = sys.argv[2]
        file_version = sys.argv[3]
        input_path = sys.argv[4]
        learning_rate = float(sys.argv[5])
        num_steps = int(sys.argv[6])
        wd = float(sys.argv[7])
        net = sys.argv[8]
        balance = sys.argv[9]
        file2load = sys.argv[10]  # which datasets to load
        load_model = sys.argv[11]  # if "Yes": use a pre-trained model
        model2load = sys.argv[12]  # which pre-trained model to use
        folds=sys.argv[13]
        test_together=sys.argv[14] # in case you choose to train the model on the combined dataset, if True test the model on the 2 cohorts together, else test each cohort seperately (still train on all the data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for fold in range(int(folds)):
            fold=str(fold)
            if load_model == "Yes":
                model = getattr(gait_segmentation, net)().to(device)
                model.load_state_dict(torch.load(os.path.join(input_path + '/' + data_type + '/model/' + model2load + "/trained_model" + model2load +'_fold_'+str(fold)+ '.pt')))
            else:
                model = getattr(gait_segmentation, net)().to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

            if os.path.isdir(os.path.join(input_path + '/' + data_type + '/model/' + file_version)) == 0:
                os.mkdir(os.path.join(input_path + '/' + data_type + '/model/' + file_version))  # open new directory if not existed

            if data_type=="all":
                if test_together:
                    train_loader, test_loader = load_dataCV(input_path, data_type, file2load,fold)
                    model, all_train_loss,all_test_loss,train_acc,validation_acc,validation_outputs =gait_segmentation.train_model_loop(model, optimizer, num_steps, train_loader, test_loader, balance)
                    with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/validation_accuracy' + data_type + file_version + '_fold_' + str(fold) + '.npy'), 'wb') as f:
                        pickle.dump(validation_acc, f, protocol=4)
                    with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/validation_outputs' + data_type + file_version + '_fold_' + str(fold) + '.npy'), 'wb') as f:
                        pickle.dump(validation_outputs, f, protocol=4)
                else:
                    train_loader, test_loader_pd,test_loader_hc = load_dataCVall(input_path, data_type, file2load,fold)
                    model, all_train_loss,train_acc, validation_acc_pd,validation_acc_hc,validation_outputs_pd,validation_outputs_hc= gait_segmentation.train_model_loop_all(model, optimizer, num_steps, train_loader, test_loader_pd, test_loader_hc, balance)
                    with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/validation_accuracy_pd' + file_version + '_fold_' + fold + '.npy'),'wb') as f:
                        pickle.dump(validation_acc_pd, f, protocol=4)
                    with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/validation_accuracy_hc' + file_version + '_fold_' + fold + '.npy'),'wb') as f:
                        pickle.dump(validation_acc_hc, f, protocol=4)
                    with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/validation_outputsPD' + file_version + '_fold_' + str(fold) + '.npy'), 'wb') as f:
                        pickle.dump(validation_outputs_pd, f, protocol=4)
                    with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/validation_outputsHC' + file_version + '_fold_' + str(fold) + '.npy'), 'wb') as f:
                        pickle.dump(validation_outputs_hc, f, protocol=4)
            else:
                train_loader, test_loader = load_dataCV(input_path, data_type, file2load, fold)
                model, all_train_loss,all_test_loss,train_acc,validation_acc,validation_outputs=gait_segmentation.train_model_loop(model, optimizer, num_steps, train_loader, test_loader, balance)
                with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/validation_accuracy' + data_type + file_version + '_fold_' + str(fold) + '.npy'), 'wb') as f:
                    pickle.dump(validation_acc, f, protocol=4)
                with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/validation_outputs' + data_type + file_version + '_fold_' + str(fold) + '.npy'), 'wb') as f:
                    pickle.dump(validation_outputs, f, protocol=4)
                with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/validation_loss' + data_type + file_version + '_fold_' + str(fold) + '.npy'), 'wb') as f:
                    pickle.dump(all_test_loss, f, protocol=4)
            with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/train_accuracy_' +data_type+ file_version + '_fold_' + fold + '.npy'),'wb') as f:
                pickle.dump(train_acc, f, protocol=4)
            with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/train_loss' + data_type + file_version + '_fold_' + str(fold) + '.npy'), 'wb') as f:
                pickle.dump(all_train_loss, f, protocol=4)
            torch.save(model.state_dict(), os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/trained_model' + file_version + '_fold_' + fold + '.pt'))

            # if data_type == 'physionet':
            #      with open(os.path.join(input_path + '/model/loss' + file_version + '.npy'),'wb') as f:
            #          pickle.dump(loss, f, protocol=4)
            #      with open(os.path.join(input_path + '/model/train_accuracy' + file_version + '.npy'),'wb') as f:
            #          pickle.dump(train_acc, f, protocol=4)
            #     with open(os.path.join(input_path + '/model/validation_accuracy' + file_version + '.npy'),'wb') as f:
            #          pickle.dump(validation_acc, f, protocol=4)
            #      torch.save(model.state_dict(),os.path.join(input_path + '/model/trained_model' + file_version + '.pt'))

    elif mode == "preds":
        data_type = sys.argv[2]
        input_path = sys.argv[3]
        file_version = sys.argv[4]
        net=sys.argv[5]
        CV=sys.argv[6]
        folds=sys.argv[7]
        file2load = sys.argv[8]  # which datasets to load, in case that you want to load dataset that processed previously
        test_together=sys.argv[9]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = getattr(gait_segmentation, net)().to(device)

        if data_type == 'physionet':
            model.load_state_dict(torch.load(os.path.join(input_path + '/model/trained_model' + file_version + '.pt')))
            test_loader=load_data(input_path,data_type,file_version)
            all_outputs = gait_segmentation.preds_append(model, test_loader)
            with open(os.path.join(input_path + '/model/all_preds' + file_version + '.npy'), 'wb') as f:
                pickle.dump(all_outputs, f, protocol=4)

        else:
            if CV=="Yes":
                preds_pd=torch.empty(0,)
                preds_hc=torch.empty(0,)
                preds_final=torch.empty(0,)
                for fold in range(int(folds)):
                    if data_type=="all" and test_together==False:
                        model.load_state_dict(torch.load(os.path.join(input_path + '/' + data_type + '/model/' + file_version + "/trained_model" + file_version + '_fold_' + str(fold) + '.pt')))
                        all_outputs_pd = open(os.path.join(input_path + '/' + data_type + '/model/' + file_version +'/validation_outputsPD'+file_version+'_fold_'+str(fold)+'.npy'),"rb")
                        all_outputs_pd = pickle.load(all_outputs_pd)
                        all_outputs_hc = open(os.path.join(input_path + '/' + data_type + '/model/' + file_version +'/validation_outputsHC'+file_version+'_fold_'+str(fold)+'.npy'),"rb")
                        all_outputs_hc = pickle.load(all_outputs_hc)
                        preds_pd = torch.cat((preds_pd, all_outputs_pd.view(-1).cpu()), 0)
                        preds_hc = torch.cat((preds_hc, all_outputs_hc.view(-1).cpu()), 0)
                        with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/all_preds_pd' + file_version + '.npy'),'wb') as f:
                            pickle.dump(preds_pd, f, protocol=4)
                        with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/all_preds_hc' + file_version + '.npy'),'wb') as f:
                            pickle.dump(preds_hc, f, protocol=4)
                    else:
                        all_outputs = open(os.path.join(input_path + '/' + data_type + '/model/' + file_version +'/validation_outputs'+data_type+file_version+'_fold_'+str(fold)+'.npy'),"rb")
                        all_outputs = pickle.load(all_outputs)
                        preds_final=torch.cat((preds_final, all_outputs.view(-1).cpu()),0)
                        with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/all_preds' + file_version + '.npy'),'wb') as f:
                            pickle.dump(preds_final, f, protocol=4)

            else:
                train_loader, test_loader = load_data(input_path, data_type, file2load)
                model.load_state_dict(torch.load(os.path.join(input_path + '/' + data_type + '/model/' + file_version + "/trained_model" + file_version + '.pt')))
                all_outputs = gait_segmentation.preds_append(model, test_loader)
                with open(os.path.join(input_path + '/' + data_type + '/model/' + file_version + '/all_preds' + file_version + '.npy'),'wb') as f:
                    pickle.dump(all_outputs, f, protocol=4)

    elif mode == "roc":
        data_type = sys.argv[2]
        file_version = sys.argv[3]
        input_path = sys.argv[4]
        CV=sys.argv[5]
        file2load=sys.argv[6]
        gait_segmentation.precision_recall(input_path, data_type, file_version,file2load,CV)

    elif mode=="post_processing":
        data_type = sys.argv[2]
        file_version = sys.argv[3]
        input_path = sys.argv[4]
        file2load=sys.argv[5] #labels location
        win = int(sys.argv[6])  # sec of minimal gait window for analysis
        fs = int(sys.argv[7])  # sampling rate
        preds_thr=float(sys.argv[8])
        gait_segmentation.post_processing(input_path,data_type,file_version,file2load,win,fs,preds_thr)

    elif mode=="daily_activity":
        data_type = sys.argv[2]
        file_version = sys.argv[3]
        input_path = sys.argv[4]
        file2load=sys.argv[5] #labels location
        preds_thr=float(sys.argv[6]) #threshold that determine the value to round the preds (values= between 0-1) high value supposed to lead to high precision and low value to high recall
        subject_level=sys.argv[7] # True= calculate the correlation in the subject level, False= in day level
        gait_segmentation.daily_activity_corr(input_path, data_type, file_version, file2load,preds_thr,subject_level)

    elif mode== "physio_exp":
        from sklearn.model_selection import LeaveOneGroupOut
        from torch.utils.data import TensorDataset, DataLoader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 64
        # kf = KFold(n_splits=3)
        physionetData = open(r'N:\Gait-Neurodynamics by Names\Yonatan\DL folders\Physionet Data\Input data\data_physionet.npy', "rb")
        physionetData = pickle.load(physionetData)
        physionetLabels = open(r'N:\Gait-Neurodynamics by Names\Yonatan\DL folders\Physionet Data\Input data\labels_physionet.npy', "rb")
        physionetLabels = pickle.load(physionetLabels)
        physionetParticipants = open(r'N:\Gait-Neurodynamics by Names\Yonatan\DL folders\Physionet Data\Input data\participants_physionet.npy',"rb")
        physionetParticipants = pickle.load(physionetParticipants)

        logo = LeaveOneGroupOut()
        count = 0
        for train_index, test_index in logo.split(physionetLabels.reshape(-1), groups=physionetParticipants.reshape(-1)):
            X_train=np.zeros((int(train_index.shape[0]/512),1,3,512))
            X_test = np.zeros((int(test_index.shape[0]/512), 1, 3, 512))
            for ind in range(3):
                physionetDataN = physionetData[:, :, ind, :].reshape(-1)

                selectedPhysioTrain = physionetDataN[train_index].reshape(int(len(train_index)/512),1,512)
                selectedPhysioTest = physionetDataN[test_index].reshape(int(len(test_index)/512),1,512)
                X_train[:,:,ind,:]=selectedPhysioTrain
                X_test[:, :, ind, :] = selectedPhysioTest

            y_train, y_test = physionetLabels.reshape(-1)[train_index], physionetLabels.reshape(-1)[test_index]
            y_train, y_test=y_train.reshape(int(len(y_train)/512),512), y_test.reshape(int(len(y_test)/512),512)
            tensor_x_train = torch.Tensor(X_train).float().to(device)  # transform to torch tensor
            tensor_y_train = torch.Tensor(y_train).float().to(device)

            train_dataset = TensorDataset(tensor_x_train, tensor_y_train)  # create your datset
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # create your dataloader

            tensor_x_test = torch.Tensor(X_test).float().to(device)  # transform to torch tensor
            tensor_y_test = torch.Tensor(y_test).float().to(device)

            test_dataset = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
            test_dataloader = DataLoader(test_dataset)  # create your dataloader
            model = gait_segmentation.Network().to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            num_steps = 20
            model, all_train_loss, all_test_loss, train_acc, validation_acc, validation_outputs = gait_segmentation.train_model_loop(
                model, optimizer, num_steps, train_dataloader, test_dataloader, 0)
            with open(os.path.join(r'N:\Gait-Neurodynamics by Names\Yonatan\DL folders\Physionet Data\Input data\model\loo\performance' + str(count) + '.npy'), 'wb') as f:
                pickle.dump(validation_acc, f, protocol=4)
            with open(os.path.join(r'N:\Gait-Neurodynamics by Names\Yonatan\DL folders\Physionet Data\Input data\model\loo\all_preds' + str(count) + '.npy'), 'wb') as f:
                pickle.dump(validation_outputs, f, protocol=4)
            count += 1


if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

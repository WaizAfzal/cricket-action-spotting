#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 13:53:27 2018

@author: Arpan

@Description: Train an RNN on C3D FC7 features for cricket stroke localization 
in untrimmed highlight videos.
"""

import os
import torch
import numpy as np
import torch.nn as nn
import pickle
import json
import time
import sys

if os.path.exists("/home/arpan/VisionWorkspace/Cricket/localization_finetuneC3D"):
    sys.path.insert(1, "/home/arpan/VisionWorkspace/Cricket/localization_finetuneC3D")
else:
    sys.path.insert(1, "/home/arpan/VisionWorkspace/localization_finetuneC3D")

import utils
import argparse
from math import fabs
from torch.utils.data import DataLoader
from Video_Dataset import VideoDataset
from model_gru import RNNClassifier
from model_gru import LSTMModel
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from get_localizations import getScoredLocalizations
from eval_shot_predictions import calculate_tIoU

c3dWinSize = 17

# Local Paths
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
#MAIN_DATASET = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_train_set"
#MAIN_LABELS = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_train_set_labels"
#VAL_DATASET = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_val_set"
#VAL_LABELS = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_val_set_labels"
c3dFC7FeatsPath = "/home/arpan/VisionWorkspace/Cricket/localization_gru/c3dFinetuned_feats_"+str(c3dWinSize)
#c3dFC7MainFeatsPath = "/home/arpan/VisionWorkspace/Cricket/localization_gru/c3dFinetunedOnHLMainSeq23_mainDataset_train_feats_"+str(c3dWinSize)
#c3dFC7ValFeatsPath = "/home/arpan/VisionWorkspace/Cricket/localization_gru/c3dFinetunedOnHLMainSeq23_mainDataset_test_feats_"+str(c3dWinSize)
base_name = "/home/arpan/VisionWorkspace/Cricket/localization_gru/logs/GRU_c3dFine_log_hidden1k_ep30_"+str(c3dWinSize)

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"
#    MAIN_DATASET = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_train_set"
#    MAIN_LABELS = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_train_set_labels"
#    VAL_DATASET = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_test_set"
#    VAL_LABELS = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_test_set_labels"
    c3dFC7FeatsPath = "/home/arpan/DATA_Drive/Cricket/extracted_feats/c3dFinetuned_feats_"+str(c3dWinSize)
#    c3dFC7MainFeatsPath = "/home/arpan/DATA_Drive/Cricket/extracted_feats/c3dFinetunedOnHLMainSeq23_mainDataset_train_feats_"+str(c3dWinSize)
#    c3dFC7ValFeatsPath = "/home/arpan/DATA_Drive/Cricket/extracted_feats/c3dFinetunedOnHLMainSeq23_mainDataset_test_feats_"+str(c3dWinSize)
    base_name = "/home/arpan/VisionWorkspace/localization_gru/logs/GRU_c3dFine_log_hidden1k_ep30_"+str(c3dWinSize)


# takes a model to train along with dataset, optimizer and criterion
def train(trainFeats, model, datasets_loader, optimizer, scheduler, criterion, \
          c3dWinSize, SEQ_SIZE, nEpochs, use_gpu, base_name):
    global training_stats
    training_stats = defaultdict()
#    best_model_wts = copy.deepcopy(model.state_dict())
#    best_acc = 0.0
    sigm = nn.Sigmoid()
    for epoch in range(nEpochs):
        print("-"*60)
        print("Epoch -> {} ".format((epoch+1)))
        training_stats[epoch] = {}
        # for each epoch train the model and then evaluate it
        for phase in ['train']:
            #print("phase->", phase)
            dataset = datasets_loader[phase]
            training_stats[epoch][phase] = {}
            accuracy = 0
            net_loss = 0
            if phase == 'train':
                scheduler.step()
                model.train(True)
            elif phase == 'test':
                #print("validation")
                model.train(False)
            
            for i, (keys, seqs, labels) in enumerate(dataset):

                # return a BATCH x (SEQ_SIZE-15) x 1 x 4096 
                if phase == 'train':
                    batchFeats = utils.getC3DFeatures(trainFeats, keys, seqs, c3dWinSize)
#                elif phase == 'test':
#                    batchFeats = utils.getC3DFeatures(valFeats, keys, seqs)
                
                # return the torch.Tensor values for inputs and 
                x, y = utils.make_c3d_variables(batchFeats, labels, c3dWinSize, use_gpu)
                #print("x type", type(x.data))
                
                preds = model(x)
                preds = sigm(preds.view(preds.size(0)))
                loss = criterion(preds, y)
                #print(preds, y)
                if torch.__version__ == '1.0.0':
                    net_loss += loss.item()
                else:
                    net_loss += loss.data[0]
                # num_ft_vecs sent to RNN = SEQ_SIZE - 15
                accuracy += get_accuracy(preds, y, (SEQ_SIZE - (c3dWinSize-1)))
#                print("# Accurate : {}".format(accuracy))
                
#                print("Phase : {} :: Batch : {} :: Loss : {} :: Accuracy : {}"\
#                          .format(phase, (i+1), net_loss, accuracy))
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
#                if (i+1) == 500:
#                    break
            accuracy = fabs(accuracy)/(len(datasets_loader[phase].dataset))
            #accuracy = fabs(accuracy)/(BATCH_SIZE*(i+1))
            training_stats[epoch][phase]['loss'] = net_loss
            training_stats[epoch][phase]['acc'] = accuracy
            training_stats[epoch][phase]['lr'] = optimizer.param_groups[0]['lr']
            
        # Display at end of epoch
        print("Phase : Train :: Epoch : {} :: Loss : {} :: Accuracy : {} : LR : {}"\
              .format((epoch+1), training_stats[epoch]['train']['loss'],\
                      training_stats[epoch]['train']['acc'], \
                                    optimizer.param_groups[0]['lr']))
#        print("Phase : Test :: Epoch : {} :: Loss : {} :: Accuracy : {}"\
#              .format((epoch+1), training_stats[epoch]['test']['loss'],\
#                      training_stats[epoch]['test']['acc']))
        
        if ((epoch+1)%nEpochs) == 0:
            save_model_checkpoint(base_name, model, epoch+1, "Adam", \
                                  win=SEQ_SIZE, use_gpu=use_gpu)

    # Save dictionary after all the epochs
    loss_filename = os.path.join(base_name, \
    "losses_GRU_c3dFC7_ep"+str(epoch+1)+"_seq"+str(SEQ_SIZE)+"_Adam.pkl")
    with open(loss_filename, 'wb') as fr:
        pickle.dump(training_stats, fr, protocol=pickle.HIGHEST_PROTOCOL)
    # Training finished


def predict(featuresPath, val_lst, classifier, val_loader, c3dWinSize,\
          use_gpu):
    
    val_keys = []
    predictions = []
    sigm = nn.Sigmoid()
    print("Loading validation/test features from disk...")
    #OFValFeatures = utils.readAllOFfeatures(OFfeaturesPath, test_lst)
    #HOGValFeatures = utils.readAllHOGfeatures(HOGfeaturesPath, val_lst)
    valFeatures = utils.readAllPartitionFeatures(featuresPath, val_lst)
    
    print("Predicting on the validation/test videos...")
    for i, (keys, seqs, labels) in enumerate(val_loader):
        
        # Testing on the sample
        #feats = getFeatureVectors(DATASET, keys, seqs)      # Parallelize this
        #batchFeats = utils.getFeatureVectorsFromDump(OFValFeatures, keys, seqs, motion=True)
        #batchFeats = utils.getFeatureVectorsFromDump(HOGValFeatures, keys, seqs, motion=False)
        batchFeats = utils.getC3DFeatures(valFeatures, keys, seqs, c3dWinSize)
        #break
        # Validation stage
        inputs, target = utils.make_c3d_variables(batchFeats, labels, c3dWinSize, use_gpu)
        #inputs, target = utils.make_variables(batchFeats, labels, motion=False)
        output = classifier(inputs) # of size (BATCHESx(SeqLen-15)) X 1

        #pred = output.data.max(1, keepdim=True)[1]  # get max value in each row
        pred_probs = sigm(output.view(output.size(0))).data  # get the normalized values (0-1)
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        val_keys.append(keys)
        predictions.append(pred_probs)  # append the 
        
        #if i % 2 == 0:
        #    print('i: {} :: Val keys: {} : seqs : {}'.format(i, keys, seqs)) #keys, pred_probs))
        #if (i+1) % 10 == 0:
        #    break
    print("Predictions done on validation/test set...")
    return val_keys, predictions
    

def save_model_checkpoint(base_name, model, ep, opt, win=16, use_gpu=True):
    """
    TODO: save the optimizer state with epoch no. along with the weights
    """
    # Save only the model params
    name = os.path.join(base_name, "GRU_c3dFC7_ep"+str(ep)+"_seq"+str(win)+"_"+opt+".pt")
#    if use_gpu and torch.cuda.device_count() > 1:
#        model = model.module    # good idea to unwrap from DataParallel and save

    torch.save(model.state_dict(), name)
    print("Model saved to disk... {}".format(name))
    

def get_accuracy(preds, targets, num_fts):
    """
    preds: a BATCH x SEQ_SIZE tensor of predictions ( sigmoid(model(x)) )
    targets: a BATCH x SEQ_SIZE tensor of binary ground truths
    ft_seq_size = SEQ_SIZE - #frames to C3D (eg. 16) + 1
                no. of C3D feat vecs sent to the RNN in the sequence
    """
    p_new = np.where(preds.data.cpu().numpy() > 0.5, 1, 0).ravel()
    #preds_new = get_1D_preds(preds)
    t_new = targets.data.cpu().numpy()
    # print("preds", preds_new[:5])
    # print("targets", tar_new[:5])
    this_batch_size = int(t_new.shape[0]/num_fts)
    th = num_fts/2.
    t_new= [1 if sum(t_new[(num_fts*i): (num_fts*(i+1))])>th else 0 \
     for i in range(this_batch_size)]
    p_new= [1 if sum(p_new[(num_fts*i): (num_fts*(i+1))])>th else 0 \
     for i in range(this_batch_size)]
    #acc = sum(p_new == t_new)*1.0
    acc = sum(np.equal(np.array(p_new), np.array(t_new))) * 1.0
    return acc

# All video files present at the same path (assumed)
def get_main_dataset_files(datasetPath):
    vfiles = sorted(os.listdir(datasetPath))         # read the filename
    return vfiles

def main(base_name, c3dWinSize=16, SEQ_SIZE=16, \
         BATCH_SIZE=256, HIDDEN_SIZE=1000, N_EPOCHS=30, N_LAYERS=1, threshold=0.5,\
         seq_threshold=0.5, use_gpu=False):
    """
    Function to read c3d FC7 features and train an RNN on them, evaluate on the 
    validation videos, write the predictions in a JSON, save the trained model and
    losses to pickle. 
    
    Parameters:
    ------
    
    base_name: path to the wts, losses, predictions and log files
    SEQ_SIZE: No. of frames sent to the RNN at a time
    BATCH_SIZE: Depends on GPU memory
    HIDDEN_SIZE: Size of hidden layer in the RNN
    N_EPOCHS: Training iterations (no. of times the training set is seen)
    N_LAYERS: No. of hidden layers in the RNN
    threshold and seq_threshold: threshold values during prediction
    use_gpu: True if training to be done on GPU, False for CPU
    
    """

    if not os.path.exists(base_name):
        os.makedirs(base_name)
    
    seed = 1234
    utils.seed_everything(seed)
    
    print(60*"#")
    
    # Form dataloaders 
#    train_lst_main_ext = get_main_dataset_files(MAIN_DATASET)   #with extensions
#    train_lst_main = [t.rsplit('.', 1)[0] for t in train_lst_main_ext]   # remove the extension
#    val_lst_main_ext = get_main_dataset_files(VAL_DATASET)
#    val_lst_main = [t.rsplit('.', 1)[0] for t in val_lst_main_ext]
    
    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
    print("c3dWinSize : {} :: SEQ_SIZE : {}".format(c3dWinSize, SEQ_SIZE))
    
    # form the names of the list of label files, should be at destination 
    train_lab = [f+".json" for f in train_lst]
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
#    train_lab_main = [f+".json" for f in train_lst_main]
#    val_lab_main = [f+".json" for f in val_lst_main]
    
    # get complete path lists of label files
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
#    tr_labs_main = [os.path.join(MAIN_LABELS, f) for f in train_lab_main]
#    val_labs_main = [os.path.join(VAL_LABELS, f) for f in val_lab_main]
    #####################################################################
    
    sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in train_lst]
    val_sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in val_lst]
#    sizes_main = [utils.getNFrames(os.path.join(MAIN_DATASET, f)) for f in train_lst_main_ext]
#    val_sizes_main = [utils.getNFrames(os.path.join(VAL_DATASET, f)) for f in val_lst_main_ext]
    
    ###########################################################################
    # Merge the training highlights and main dataset variables
#    train_lab.extend(train_lab_main)
#    tr_labs.extend(tr_labs_main)
#    sizes.extend(sizes_main)
    
    print("No. of training videos : {}".format(len(train_lst)))
    
    print("Size : {}".format(sizes))
    hlDataset = VideoDataset(tr_labs, sizes, seq_size=SEQ_SIZE, is_train_set = True)
    print(hlDataset.__len__())
    
    #####################################################################
    
    # Create a DataLoader object and sample batches of examples. 
    # These batch samples are used to extract the features from videos parallely
    train_loader = DataLoader(dataset=hlDataset, batch_size=BATCH_SIZE, shuffle=True)
    datasets_loader = {'train': train_loader}       # can have a test loader also
    
    # read into dictionary {vidname: np array, ...}
    print("Loading features from disk...")
    #HOGfeatures = utils.readAllHOGfeatures(HOGfeaturesPath, train_lst)
    features = utils.readAllPartitionFeatures(c3dFC7FeatsPath, train_lst)
#    mainFeatures = utils.readAllPartitionFeatures(c3dFC7MainFeatsPath, train_lst_main)
#    features.update(mainFeatures)     # Merge dicts
    print(len(train_loader.dataset))
#    
#    ########
#    
#    #fc7 layer output size
#    INP_VEC_SIZE = features[list(features.keys())[0]].shape[-1] 
    INP_VEC_SIZE = 4096
    print("INP_VEC_SIZE = ", INP_VEC_SIZE)
    
    # Creating the RNN and training
    classifier = RNNClassifier(INP_VEC_SIZE, HIDDEN_SIZE, 1, N_LAYERS, \
                               bidirectional=False, use_gpu=use_gpu)
#    classifier = LSTMModel(INP_VEC_SIZE, HIDDEN_SIZE, 1, N_LAYERS, \
#                           use_gpu=use_gpu)
    if use_gpu:
#        if torch.cuda.device_count() > 1:
#            print("Let's use", torch.cuda.device_count(), "GPUs!")
#            # Parallely run on multiple GPUs using DataParallel
#            classifier = nn.DataParallel(classifier)
        classifier.cuda(0)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    #criterion = nn.CrossEntropyLoss()
    
    criterion = nn.BCELoss()

    step_lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    start = time.time()
    
#    print("Training for %d epochs..." % N_EPOCHS)
#    # Training the model on the features for N_EPOCHS 
    train(features, classifier, datasets_loader, optimizer, step_lr_scheduler, \
          criterion, c3dWinSize, SEQ_SIZE, N_EPOCHS, use_gpu, base_name)
    mod_file = os.path.join(base_name, \
                "GRU_c3dFC7_ep"+str(N_EPOCHS)+"_seq"+str(SEQ_SIZE)+"_Adam.pt")
    classifier.load_state_dict(torch.load(mod_file))    
    end = time.time()
    print("Time for training : {}".format(end-start))
    #####################################################################
    
    # Test a video or calculate the accuracy using the learned model
    print("Prediction video meta info.")
#    print("Size : {}".format(val_sizes_main))
    hlvalDataset = VideoDataset(val_labs, val_sizes, seq_size=SEQ_SIZE, \
                                is_train_set = False)
    print(hlvalDataset.__len__())
    
    # Create a DataLoader object and sample batches of examples. 
    # These batch samples are used to extract the features from videos parallely
    val_loader = DataLoader(dataset=hlvalDataset, batch_size=BATCH_SIZE, shuffle=False)
    #print(len(val_loader.dataset))

    classifier.eval()
    val_keys, predictions = predict(c3dFC7FeatsPath, val_lst, classifier, val_loader, \
          c3dWinSize, use_gpu)
    
    with open(os.path.join(base_name, "predictions_seq"+str(SEQ_SIZE)+".pkl"), "wb") as fp:
        pickle.dump(predictions, fp)
    
    with open(os.path.join(base_name, "val_keys_seq"+str(SEQ_SIZE)+".pkl"), "wb") as fp:
        pickle.dump(val_keys, fp)

    #####################################################################

    # [4949, 4369, 4455, 4317, 4452]
    #predictions = [p.cpu() for p in predictions]  # convert to CPU tensor values
    localization_dict = getScoredLocalizations(val_keys, predictions, BATCH_SIZE, \
                                         threshold, seq_threshold)

#    print localization_dict

    # Apply filtering    
    i = 60  # optimum
    filtered_shots = utils.filter_action_segments(localization_dict, epsilon=i)
    #i = 7  # 
    #filtered_shots = filter_non_action_segments(filtered_shots, epsilon=i)
    filt_shots_filename = os.path.join(base_name, "predicted_localizations_mainTest_th0_5_filt"\
            +str(i)+"_ep"+str(N_EPOCHS)+"_seq"+str(SEQ_SIZE)+".json")
    with open(filt_shots_filename, 'w') as fp:
        json.dump(filtered_shots, fp)
    print("Prediction file {} !!".format(filt_shots_filename))
    
    tiou =  calculate_tIoU(LABELS, filtered_shots)
    #####################################################################
    # count no. of parameters in the model
    print("#Parameters : {} ".format(utils.count_parameters(classifier)))
    
    print(60*'#')
    return tiou


if __name__=='__main__':
    
    SEQ_SIZE = 16   # has to >=c3dWinSize (ie. the number of frames used for c3d input)
    BATCH_SIZE = 256
    # Parameters and DataLoaders
    HIDDEN_SIZE = 1000
    N_EPOCHS = 30
    N_LAYERS = 1        # no of hidden layers
    threshold = 0.5
    seq_threshold = 0.5
    use_gpu = torch.cuda.is_available()
    #use_gpu = True
    
    description = "Script for training RNNs on C3D FC7 features"
    p = argparse.ArgumentParser(description=description)
    
#    p.add_argument('-ds', '--DATASET', type=str, default=DATASET,
#                   help=('input directory containing input videos'))
#    p.add_argument('-labs', '--LABELS', type=str, default=LABELS,
#                   help=('output directory for c3d feats'))
#    p.add_argument('-feats', '--featuresPath', type=str, default=c3dFC7FeatsPath,
#                   help=('extracted c3d FC7 features path'))
    p.add_argument('-dest', '--base_name', type=str, default=base_name, 
                   help=('wts, losses and log file path'))
    
    p.add_argument('-c3dwin', '--c3dWinSize', type=int, default=c3dWinSize)
    p.add_argument('-w', '--SEQ_SIZE', type=int, default=SEQ_SIZE)
    p.add_argument('-b', '--BATCH_SIZE', type=int, default=BATCH_SIZE)
    p.add_argument('-hs', '--HIDDEN_SIZE', type=int, default=HIDDEN_SIZE)
    p.add_argument('-n', '--N_EPOCHS', type=int, default=N_EPOCHS)
    p.add_argument('-l', '--N_LAYERS', type=int, default=N_LAYERS)    
    p.add_argument('-t', '--threshold', type=int, default=threshold)
    p.add_argument('-s', '--seq_threshold', type=int, default=seq_threshold)
    p.add_argument('-g', '--use_gpu', type=bool, default=use_gpu)
    
    # create dictionary of tiou values and save to destination 
    tiou_dict = {}
    
    for seq in range(30, 36):
        p.set_defaults(SEQ_SIZE = seq)
        tiou = main(**vars(p.parse_args()))
        tiou_dict[seq] = tiou
    
    dest = vars(p.parse_args())['base_name']
#    with open(os.path.join(dest, 'tiou'+'_HD'+str(HIDDEN_SIZE)+'.json'), 'w') as fp:
#        json.dump(tiou_dict, fp)
    print("TIoU values written for all iterations !!")
    
#    #####################################################################


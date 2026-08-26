#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 14:06:55 2019

Description: Script to plot the GRU network accuracies and losses

@author: arpan
"""


import numpy as np
import matplotlib.pyplot as plt
#import mmap
#import glob
#import re
from collections import defaultdict
import os
import json
import sys

if os.path.exists("/home/arpan/VisionWorkspace/Cricket/localization_finetuneC3D"):
    sys.path.insert(1, "/home/arpan/VisionWorkspace/Cricket/localization_finetuneC3D")
else:
    sys.path.insert(1, "/home/arpan/VisionWorkspace/localization_finetuneC3D")
    
from get_localizations import getScoredLocalizations
from eval_shot_predictions import calculate_tIoU


LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"

#import seaborn as sns
#
#sns.set()

def plot_accuracy(d, xlab, ylab):
    keylist = d.keys()
    keylist.sort()
    val_list = []
    for key in keylist:
        val_list.append(d[key])
    print("Iteration and Accuracy Lists : ")
    print(keylist)
    print(val_list)
    plt.figure(2)
    plt.title("Accuracy Vs Iteration for validation set")    
    plt.plot(keylist, val_list, lw=1, color='b', marker='.')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.show()
    plt.savefig('accuracy.png', bbox_inches='tight')    
    return

def plot_accuracy_BGThresh(x, y, xlab, ylab):
    
    plt.figure(2)
    plt.title("Weighted Mean TIoU Vs Segment size", fontsize=16)
    plt.plot(x, y, lw=1, color='b', marker='.')
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    plt.axvline(x=60, color = 'r', linestyle = '--')
    plt.legend()
    plt.show()
    plt.savefig('tiou_filter60.png', bbox_inches='tight')    
    return

def plot_losses(tr_loss, test_loss, xlab, ylab):
    tr_keylist = tr_loss.keys()
    test_keylist = test_loss.keys()
    tr_keylist.sort()
    test_keylist.sort()
    tr_val_list = []
    test_val_list = []
    for key in tr_keylist:
        tr_val_list.append(tr_loss[key])
    for key in test_keylist:
        test_val_list.append(test_loss[key])

    print("Iteration and Training Loss Lists : ")
    print(tr_keylist)
    print(tr_val_list)
    print("Iteration and Testing Loss Lists : ")
    print(test_keylist)
    print(test_val_list)    
    
    plt.figure(1)
    plt.title("Loss Vs Iteration for training and validation set", fontsize=16)
    #plt.title("")
    plt.plot(tr_keylist, tr_val_list, lw=1, marker='.',color='g', label='Training Loss')
    plt.plot(test_keylist, test_val_list, lw=1, marker='.', color='r', label='Validation Loss')
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    plt.legend()
    plt.show()
    plt.savefig('c3d_losses.png', bbox_inches='tight')
    return    

def plot_train_loss(keys, l, xlab, ylab):
    
    keylist = range(1,91)      # x-axis for 100 epochs
    cols = ['r','g','b', 'c']        
        
    print("Iteration and Accuracy Lists : ")
    print(keylist)
    print(l)
    plt.figure(2)
    plt.title("Training Loss Vs Epoch (Batch=256)")    
    for i in range(len(keys)):
        plt.plot(keylist, l[keys[i]], lw=1, color=cols[i], marker='.', label= keys[i])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.show()
    plt.savefig('gru_losses.png', bbox_inches='tight')    
    return

def plot_accuracy_OFG30(x, y, xlab, ylab):
    cols = ['r','g','b', 'c']        
    plt.figure(2)
    plt.title("Weighted Mean TIoU Vs SeqLen", fontsize=16)
    plt.plot(x, y, lw=1, color='b', marker='.')
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    #plt.axvline(x=60, color = 'r', linestyle = '--')
    #plt.legend()
    plt.show()
    plt.savefig('OFGrid30_accuracies.png', bbox_inches='tight')    
    return

def plot_accuracy_gru_w17_23(x, keys, l, xlab, ylab, fname):
    
    cols = ['r','g','b', 'c']        
    print(l)
    fig = plt.figure(2)
    plt.title("Weighted Mean TIoU Vs SeqLen", fontsize=12)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc=4)
    plt.ylim(ymin=0, ymax=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def get_prediction_accuracy(base_name, filterwidth, epoch, seq):
    
    filename = "predicted_localizations_th0_5_filt"
    c3d_filename = "predicted_localizations_mainTest_th0_5_filt"
    if "c3dFine" in base_name:
        filt_shots_filename = os.path.join(base_name, c3d_filename+str(filterwidth)\
                                       +"_ep"+str(epoch)+"_seq"+str(seq)+".json")
    else:
        filt_shots_filename = os.path.join(base_name, filename+str(filterwidth)\
                                       +"_ep"+str(epoch)+"_seq"+str(seq)+".json")
    with open(filt_shots_filename, 'r') as fp:
        filtered_shots = json.load(fp)
    print("Prediction file {} !!".format(filt_shots_filename))
    tiou =  calculate_tIoU(LABELS, filtered_shots)
    return tiou


if __name__=='__main__':

    base_name = "/home/arpan/VisionWorkspace/Cricket/localization_gru/logs"
    if not os.path.exists(base_name):
        base_name = "/home/arpan/VisionWorkspace/localization_gru/logs/GRU_hogHL_log_hidden1k"
        
    hog_path = "GRU_hogHL_log_hidden1k"
    hoof_path = "GRU_hoofHL_log_hidden1k"
    of30_path = "GRU_of20HL_log_hidden1k"
    c3d_path = "GRU_c3dFine_log_hidden1k_ep30_17"
    
    N_EPOCHS = 30
    seq_vals = []
    hog_acc = []
    hoof_acc = []
    of30_acc = []
    accuracy_17 = [0.8327131315616983, 0.8380260939405374, 0.8500151106233975,
                0.8301265650686521, 0.8290406996776241, 0.8343860022862118,
                0.839508014377935, 0.8369764142766354, 0.8304449001717371,
                0.8389650619634795, 0.835555156427209, 0.8329047661506382,
                0.8251018653034401, 0.8340082508611669, 0.8368660249504486,
                0.8062165073029551, 0.7962174950378241, 0.8038797659889161,
                0.8079941495597807]    
    accuracy_17_30ep = []
    
    for seq in range(1, 36):
        seq_vals.append(seq)
        hog_acc.append(get_prediction_accuracy(os.path.join(base_name, hog_path), 60, N_EPOCHS, seq))
        if seq==1:
            continue
        hoof_acc.append(get_prediction_accuracy(os.path.join(base_name, hoof_path), 60, N_EPOCHS, seq))
        of30_acc.append(get_prediction_accuracy(os.path.join(base_name, of30_path), 60, N_EPOCHS, seq))
        if seq>=17:
            accuracy_17_30ep.append(get_prediction_accuracy(os.path.join(base_name, c3d_path), 60, N_EPOCHS, seq))
    
    keys = ["HOG", "HOOF", "OF Grid 20", "C3D $\mathit{FC7}$: $w_{c3d}=17$"]
    l = {keys[0]: hog_acc, keys[1]: hoof_acc, keys[2]: of30_acc, keys[3]:accuracy_17_30ep}
    
    plot_accuracy_gru_w17_23(seq_vals, keys, l, 'Sequence length ($\mathit{SeqLen}$)', \
                             "Weighted Mean TIoU", 'new_tiou_allFeats_1_35_new.png')
    
################################################################################
#    # For OF results    
#    file_w17_17 = "logs/GRU_c3dFine_log_hidden1k_17/losses_GRU_c3dFC7_ep90_seq17_Adam.pkl"
#    file_w17_35 = "logs/GRU_c3dFine_log_hidden1k_17/losses_GRU_c3dFC7_ep90_seq35_Adam.pkl"
#    file_w23_23 = "logs/GRU_c3dFine_log_hidden1k_23/losses_GRU_c3dFC7_ep90_seq23_Adam.pkl"
#    file_w23_35 = "logs/GRU_c3dFine_log_hidden1k_23/losses_GRU_c3dFC7_ep90_seq35_Adam.pkl"
#    
#    files = [file_w17_17, file_w17_35, file_w23_23, file_w23_35]
#    keys = ["W=17; SeqLen=17", "W=17; SeqLen=35", "W=23; SeqLen=23", "W=23; SeqLen=35"]
#    losses = {}
#    
#    for i,fname in enumerate(files):
#        with open(fname, "rb") as fp:
#            #losses[keys[i]] = pickle.load(fp)
#            stats = pickle.load(fp)
#        loss_vals = []
#        for j in range(90):
#            tr_res = stats[j]
#            loss_vals.append(tr_res['train']['loss'])
#            
#        losses[keys[i]] = loss_vals
#    
#    keylist = range(1,91)      # x-axis for 100 epochs
#    cols = ['r','g','b', 'c']        
#        
#    print("Iteration and Accuracy Lists : ")
#    print(keylist)
#    print(losses)
#    fig = plt.figure()
#    plt.title("Training Loss Vs Epoch (Batch=256)")    
#    for i in range(len(keys)):
#        if "W=23" in keys[i]:
#            plt.plot(keylist, losses[keys[i]], lw=1, color=cols[i], label= keys[i])
#        else:
#            plt.plot(keylist, losses[keys[i]], lw=1, color=cols[i], label= keys[i])
#    plt.xlabel("#Epoch")
#    plt.ylabel("Training Loss")
#    plt.legend()
#    plt.show()
#    fig.savefig('gru_losses.png', bbox_inches='tight')    
#    plt.close(fig)

################################################################################
#    # For C3D W=17 features
#    accuracy_17 = [0.8327131315616983, 0.8380260939405374, 0.8500151106233975,
#                0.8301265650686521, 0.8290406996776241, 0.8343860022862118,
#                0.839508014377935, 0.8369764142766354, 0.8304449001717371,
#                0.8389650619634795, 0.835555156427209, 0.8329047661506382,
#                0.8251018653034401, 0.8340082508611669, 0.8368660249504486,
#                0.8062165073029551, 0.7962174950378241, 0.8038797659889161,
#                0.8079941495597807]    
#    accuracy_23 = [0.741046144795, 0.777184291238, 0.779685126561, 
#                   0.786273560059, 0.787677832275, 0.796398087159,
#                   0.811448712847, 0.81133483445, 0.792795800061,
#                   0.806553073775, 0.818204811244, 0.811252896884, 
#                   0.814451161111]    
#    #accuracy_30 = [0.6198, 0.6328,  0.4708, 0.4963, 0.4560]    
#    #accuracy_20 = [0.6338, 0.5444,  0.5796 , 0.5246, 0.4820]    
#    x = range(17, 36)
#    keys = ["For W=17", "For W=23"]
#    
#    l = {keys[0]: accuracy_17, keys[1]: accuracy_23}
#    
#    #plot_accuracy_OFG30(range(2,11), accuracy, 'Sequence length (SeqLen)', "Weighted Mean TIoU")
#    plot_accuracy_gru_w17_23(x, keys, l, 'Sequence length (SeqLen)', \
#                             "Weighted Mean TIoU", 'tiou_w17_23.png')

#    ###########################################################################            

#    # Plot mAP values    
#    # For C3D W=17 features
#    accuracy_17 = [0.646165347852709, 0.6631119558329369, 0.6668565871999088,
#                   0.622168307054952, 0.6134411611423951, 0.6297086159738986,
#                   0.6350069638677267, 0.6348420319320572, 0.6135786709805868,
#                   0.6372346810329992, 0.6295973729201413, 0.6215299542493754,
#                   0.5918313319209896, 0.5897829944398387, 0.6085711550933354,
#                   0.5478008479799334, 0.5167893446833662, 0.5391605993901684,
#                   0.5339356341955489]    
#    accuracy_23 = [0.5052265074835207, 0.5586861744082112, 0.5692041766883036,
#                   0.5788523806318883, 0.5811570216790299, 0.5928631917268791,
#                   0.6121033601879127, 0.6175097497166658, 0.5704309092692883,
#                   0.5977874703471866, 0.6148702626494703, 0.5951486393030462,
#                   0.5791171192270526]
#    #accuracy_30 = [0.6198, 0.6328,  0.4708, 0.4963, 0.4560]    
#    #accuracy_20 = [0.6338, 0.5444,  0.5796 , 0.5246, 0.4820]    
#    x = range(17, 36)
#    keys = ["For W=17", "For W=23"]
#    
#    l = {keys[0]: accuracy_17, keys[1]: accuracy_23}
#    
#    #plot_accuracy_OFG30(range(2,11), accuracy, 'Sequence length (SeqLen)', "Weighted Mean TIoU")
#    plot_accuracy_gru_w17_23(x, keys, l, 'Sequence length (SeqLen)', \
#                             "mean Average Precision", 'map_w17_23.png')


#
#    ###########################################################################        
#    with open(os.path.join(cur_dir, filename+files[0]+".pickle"), 'rb') as fp:
#        stats_seq23 = pickle.load(fp)
#    
#    epochs_keys = stats_seq23.keys()
#    losses_seq23 = [stats_seq23[k]['train']["loss"][0] for k in epochs_keys]
#    accuracies_seq23 = [stats_seq23[k]['train']["acc"] for k in epochs_keys]
#    #lr = [stats[k]['train']["lr"] for k in epochs_keys]
#
#    ep_seq23 = [(k+1) for k in epochs_keys]
#    plt.plot(ep_seq23, losses_seq23, lw=1, color=cols[0], marker='.', label="For window size 23")
#    #plt.plot(ep_sc0, accuracies_sc0, lw=1, color=cols[1], marker='*', label="Accuracy "+files[1])
#    
#    ###########################################################################        
#    
#    
#    
#    plt.xlabel(xlab)
#    plt.ylabel(ylab)
#    plt.legend(prop={'size':13}) # 7 or 'center right'
#    #plt.ylim(ymin=0, ymax=1)
#    plt.show()
#    plt.savefig('c3dFine_seq23_losses.png', bbox_inches='tight')    


    
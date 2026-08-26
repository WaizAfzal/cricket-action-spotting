#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat June 29 01:34:25 2019

@author: Arpan

@Description: Utils file to extract Farneback dense optical flow features 
from folder videos and dump to disk.

Feature : Farneback Dense Optical Flow: Magnitudes and Angles (with grid_size)
Execution Time: 1365.583 secs (Njobs=10, batch=10) (nVids = 26) grid=20
Execution Time: 1420.7 secs (Njobs=10, batch=10) (nVids=26) grid=40
Execution Time: 1326.21 secs (Njobs=10, batch=10) (nVid=26) grid=30
"""

import os
import numpy as np
import cv2
import time
import sys
import pandas as pd
from joblib import Parallel, delayed

def extract_hoof_vids(srcFolderPath, destFolderPath, bins, mag_thresh=5, \
                           njobs=1, batch=10, stop='all'):
    """
    Function to extract the features from a list of videos, given the path of the
    videos and the destination path for the features
    
    Parameters:
    ------
    srcFolderPath: str
        path to folder which contains the videos
    destFolderPath: str
        path to store the optical flow values in .npy files
    threshold: int
        threshold for magnitude
    bins: np.linspace()
        bin divisions
    mag_thresh: int
        for pixels having magnitude > mag_thresh
    njobs: int
        no. of cores to be used parallely
    batch: int
        no. of video files in a batch. A batch executed parallely and 
        is dumped to disk before starting another batch. Depends on RAM.
    stop: str or int(if to be stopped after some files)
        to traversel 'stop' no of files in each subdirectory.
    
    Return: 
    ------
    traversed: int
        no of videos traversed successfully
    """
    
    # iterate over the subfolders in srcFolderPath and extract for each video 
    vfiles = os.listdir(srcFolderPath)
    
    infiles, outfiles, nFrames = [], [], []
    
    traversed = 0
    # create destination path to store the files
    if not os.path.exists(destFolderPath):
        os.makedirs(destFolderPath)
            
    # iterate over the video files inside the directory sf
    for vid in vfiles:
        if os.path.isfile(os.path.join(srcFolderPath, vid)) and vid.rsplit('.', 1)[1] in {'avi', 'mp4'}:
            infiles.append(os.path.join(srcFolderPath, vid))
            outfiles.append(os.path.join(destFolderPath, vid.rsplit('.',1)[0]+".npy"))
            nFrames.append(getTotalFramesVid(os.path.join(srcFolderPath, vid)))
            # save at the destination, if extracted successfully
            traversed += 1
                   
            # to stop after successful traversal of 2 videos, if stop != 'all'
            if stop != 'all' and traversed == stop:
                break
                    
    print("No. of files to be written to destination : "+str(traversed))
    if traversed == 0:
        print("Check the structure of the dataset folders !!")
        return traversed
    ###########################################################################
    #### Form the pandas Dataframe and parallelize over the files.
    filenames_df = pd.DataFrame({"infiles":infiles, "outfiles": outfiles, "nframes": nFrames})
    filenames_df = filenames_df.sort_values(["nframes"], ascending=[True])
    filenames_df = filenames_df.reset_index(drop=True)
    nrows = filenames_df.shape[0]
    
    for i in range(int(nrows/batch)):
        # 
        print("i = "+str(i))
        batch_diffs = Parallel(n_jobs=njobs)(delayed(extract_flow_angles) \
                          (filenames_df['infiles'][i*batch+j], \
                           [(0, filenames_df['nframes'][i*batch+j]-1)], \
                           bins, mag_thresh) for j in range(batch))
        
        # Writing the files to the disk in a serial manner
        for j in range(batch):
            if batch_diffs[j] is not None:
                np.save(filenames_df['outfiles'][i*batch+j], batch_diffs[j])
                print("Written "+str(i*batch+j+1)+" : "+ \
                                    filenames_df['outfiles'][i*batch+j])
            
    # For last batch which may not be complete, extract serially
    last_batch_size = nrows - (int(nrows/batch)*batch)
    if last_batch_size > 0:
        batch_diffs = Parallel(n_jobs=njobs)(delayed(extract_flow_angles) \
                              (filenames_df['infiles'][int(nrows/batch)*batch+j], \
                               [(0, filenames_df['nframes'][int(nrows/batch)*batch+j]-1)], \
                               bins, mag_thresh) for j in range(last_batch_size)) 
        # Writing the files to the disk in a serial manner
        for j in range(last_batch_size):
            if batch_diffs[j] is not None:
                np.save(filenames_df['outfiles'][int(nrows/batch)*batch+j], batch_diffs[j])
                print("Written "+str((nrows/batch)*batch+j+1)+" : "+ \
                                    filenames_df['outfiles'][int(nrows/batch)*batch+j])
    
    ###########################################################################
    return traversed


def getTotalFramesVid(srcVideoPath):
    """
    Return the total number of frames in the video
    Parameter:
    ------
    srcVideoPath: str
        complete path to the video file
        
    Returns: 
    ------
    tot_frames: int
        total number of frames in the video file.
    """
    cap = cv2.VideoCapture(srcVideoPath)
    # if the videoCapture object is not opened then return 0 frames
    if not cap.isOpened():
        print("Error reading the video file !!")
        return 0

    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return tot_frames

def extract_flow_angles(vidFile, frame_indx, hist_bins, mag_thresh):
    '''
    Extract optical flow maps from video vidFile for all the frames and put the angles with >mag_threshold in different 
    bins. The bins vector is the feature representation for the stroke. 
    Use only the strokes given by list of tuples frame_indx.
    Parameters:
    ------
    vidFile: str
        complete path to a video
    frame_indx: list of tuples (start_frameNo, end_frameNo)
        each tuple in the list denotes the starting frame and ending frame of a stroke.
    hist_bins: 1d np array 
        bin divisions (boundary values). Used np.linspace(0, 2*PI, 11) for 10 bins
    mag_thresh: int
        minimum size of the magnitude vectors that are considered (no. of pixels shifted in consecutive frames of OF)
    
    '''
    
    cap = cv2.VideoCapture(vidFile)
    if not cap.isOpened():
        print("Capture object not opened. Aborting !!")
        sys.exit(0)
    ret = True
    features_current_file = []
    prvs, next_ = None, None
    for m, n in frame_indx:   #localization tuples
        
        #print("stroke {} ".format((m, n)))
#        sum_norm_mag_ang = np.zeros((len(hist_bins)-1))  # for optical flow maxFrames - 1 size
        frameNo = m
        while ret and frameNo <= n:
            if (frameNo-m) == 0:    # first frame condition
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameNo)
                ret, frame1 = cap.read()
                if not ret:
                    print("Frame not read. Aborting !!")
                    break
                # resize and then convert to grayscale
                prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                #prvs = scale_and_crop(prvs, scale)
                frameNo +=1
                continue
                
            ret, frame2 = cap.read()
            # resize and then convert to grayscale
            next_ = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #tvl1_flow = cv2.DualTVL1OpticalFlow_create()
            #flow = tvl1_flow.calc(prvs, next_, None)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            
            pixAboveThresh = np.sum(mag>mag_thresh)
            #inv = np.sum(np.isnan(mag[mag>mag_thresh]))
            #print("Frame = {} :: {}".format(frameNo, pixAboveThresh))
            #use weights=mag[mag>THRESH] to be weighted with magnitudes
            #returns a tuple of (histogram, bin_boundaries)
            ang_hist = np.histogram(ang[mag>mag_thresh], bins=hist_bins)[0]#, \
#                                    weights=mag[mag>mag_thresh])[0]
#            sum_norm_mag_ang +=ang_hist[0]
#            if not pixAboveThresh==0:
#                sum_norm_mag_ang[frameNo-m-1] = np.sum(mag[mag > THRESH])/pixAboveThresh
#                sum_norm_mag_ang[(maxFrames-1)+frameNo-m-1] = np.sum(ang[mag > THRESH])/pixAboveThresh
            frameNo+=1
            prvs = next_
            hoof_feature = np.expand_dims(ang_hist.flatten(), axis = 0)
            #print("---Done---")
            features_current_file.append(hoof_feature)
        #stroke_features.append(sum_norm_mag_ang/(n-m+1))
    cap.release()
    #cv2.destroyAllWindows()
    feat_mat = np.array(features_current_file)
    feat_mat[np.isnan(feat_mat)] = 0
    return feat_mat


if __name__=='__main__':
    
    nbins = 100
    mag_thresh = 0.5
    batch = 10  # No. of videos in a single batch
    njobs = 10   # No. of threads
    # Server params
#    srcPath = '/opt/datasets/cricket/ICC_WT20'
#    destPath = "/home/arpan/DATA_Drive/Cricket/extracted_feats/HOOF_HL_bins"+str(nbins)+"_th"+str(mag_thresh)
    srcPath = '/home/arpan/DATA_Drive/Cricket/dataset_25_fps_test_set'
    destPath = "/home/arpan/DATA_Drive/Cricket/extracted_feats/HOOF_mainDataset_test_bins"+str(nbins)+"_th"+str(mag_thresh)

    # localhost params
    if not os.path.exists(srcPath):
        srcPath = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_test_set"
        destPath = "/home/arpan/VisionWorkspace/Cricket/localization_gru/HOOF_mainDataset_test_bins"+str(nbins)+"_th"+str(mag_thresh)
        
    bins = np.linspace(0, 2*np.pi, (nbins+1))
    start = time.time()
    extract_hoof_vids(srcPath, destPath, bins, mag_thresh, njobs, \
                           batch, stop='all')
    end = time.time()
    print("Total execution time : "+str(end-start))
    
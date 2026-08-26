# localization_gru
A repository for localization of Cricket strokes on pre - extracted features, using GRU networks. This code requires supporting files, which can be found in localization_finetuneC3D repo.

## Shared files and folder Details:
________________________________________________________

main.py: 	main file for training GRU model and testing on Highlight video c3d features.

main_c3dMain.py: 	main file for training GRU model and testing on c3d Generic set features.

main_hog.py: 	main file for using HOG features.

main_hoof.py: 	main file for using HOOF features.

main_of.py: 	main file for using Optical flow grid features.

main_run_HL_c3d17_SEQ34_35.py: 	for given sequences.

model_gru.py: 	Model of GRU RNN defined here

extract_hof_par.py:		extract HOOF features, parallelized.

calc_plot_acc.py, plot_acc_loss_gru.py and plot_gru.py: 	for plotting from log files.

logs/ : folder containing the log dump files and the stdout from different runs.

________________________________________________________


### Useful files from localization_finetuneC3D:

c3d.pickle: 	C3D model weights file(Pretrained on Sports1M dataset). It was downloaded as is.

finetune_c3d.py: 	main file for finetuning a pretrained C3D model and dump weights to disk.

model_c3d.py: 	c3d defined and may be loaded with pretrained wts (in c3d.pickle)

eval_shot_predictions.py: 	Calculate the TIoU metric.

get_localizations.py: 	Used to create localized segments.

Video_Dataset.py: 	Subclassing Dataset class

utils.py: 	Some utility functions that are used throughout this project and other adjacent projects

extract_c3dFeats_par.py: 	To extract c3d FC7 layers features using a finetuned C3D model, parallelized.

________________________________________________________


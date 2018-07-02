# Project-ai
Adenocarcinoma classification using deep neural networks


root:
	|data -> GlaS
	|trainingg_models
	|all codes should be here

#-------------------- Data set and trained models ------------------
GlaS Dataset can be downloaded from:
https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest

Trained models can are provided in this link:

https://drive.google.com/open?id=1g6dhJ80zjqVA3hyQotxnbi2ZpLIzdlSl


#------------------------- Running instruction ---------------------

Please type this in bash for training and evaluation:
bash run_exp.sh
Then you may run this for evaluation on test set:
bash run_eval.sh

if you can not use bash you may use this command in command line for trainig a model:
python modified_UNet_test.py experiment_num loss_type learning_rate

if you can not use bash you may use this command in command line, for evaluating the trained model:
python evaluation.py experiment_num loss_type


Because we have uploaded our trained models with the experiment number of 1, in the run_exp.sh experiment_num is equal to 2 to not overwrite those files. 
If you wish to evaluate another experiment naturally you have to change experiment number in the run_eval.sh

#----------------------- Results and models -----------------------

all the models will be save in trained_models folder in a folder with a name corresponding to their loss type and experiment number
all results will be saved in the same directory in the final_results folder

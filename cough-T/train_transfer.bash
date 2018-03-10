#!/bin/bash


# All the weights got random weights
# We fine tune the top 3 layers and the bottom layer

##################################################################################
#
#	DIRECTORIES
#
##################################################################################

#directory where the dataset is stored
DATASET_DIR=Audio_Data/cough
TRANS_DATASET_DIR=Audio_Data/cough

#directory where the newly created checkpoints + the summaries are stored
NEW_CHECKPOINT_DIR=checkpoints/transf



##################################################################################
#
#	PARAMETER
#
##################################################################################


BATCH_SIZE=128

#the transfer learning dataset consists of how many classes?
NUM_T_CLASSES=2
	




##################################################################################
#
#	START PROGRAM
#
##################################################################################

mkdir -p ${NEW_CHECKPOINT_DIR}

echo '1) train top layer for 8k steps'
python controller.py  \
	--eta=1e-2  \
	--batch_size=${BATCH_SIZE} \
	--num_classes=${NUM_T_CLASSES} \
        --num_steps=5000 \
        --dataset_dir=${TRANS_DATASET_DIR} \
	--checkpoint_dir=${NEW_CHECKPOINT_DIR}/phase1 


echo '2) train top 3 layers for 15k steps'
python controller.py  \
	--eta=1e-3  \
	--batch_size=${BATCH_SIZE} \
	--num_classes=2 \
        --num_steps=5000 \
        --dataset_dir=${DATASET_DIR} \
	--checkpoint_dir=${NEW_CHECKPOINT_DIR}/phase2 \
        --restore_model_path=${NEW_CHECKPOINT_DIR}/phase1 \
        --trainable_scopes=['top']


echo '3) train all layers for 1.5k steps'
python controller.py  \
	--eta=5e-4  \
	--batch_size=${BATCH_SIZE} \
	--num_classes=2 \
        --num_steps=2000 \
        --dataset_dir=${DATASET_DIR} \
	--checkpoint_dir=${NEW_CHECKPOINT_DIR}/phase3 
        --restore_model_path=${NEW_CHECKPOINT_DIR}/phase2 \


echo '4) evaluate the model'
python controller.py  \
	--batch_size=${BATCH_SIZE} \
	--num_classes=2 \
        --num_epochs=20 \
        --dataset_dir=${DATASET_DIR} \
        --restore_model_path=${NEW_CHECKPOINT_DIR}/phase3 \
        --no_training


# Data Processing
DATASET_NAME=itc99 #itc99 openabcd forgeeda
SAMPLE_RATIO=0.01 ## 0.2 for ITC99; 0.05 for OpenABCD, 0.003 for ForgeEDA
RAW_DATA=./raw_data/$DATASET_NAME
AIG_DIR=./dataset/$DATASET_NAME/aig
PM_DIR=./dataset/$DATASET_NAME/pm
PT_DIR=./dataset/$DATASET_NAME/pm_aig
REV_DIR=./dataset/$DATASET_NAME/pm2aig
BDY_DIR=./dataset/$DATASET_NAME/boundary
NUM_WORKERS=1 # set num_workers to 1 if there is a bug
ABC_PATH=/your/abc/path


mkdir -p $AIG_DIR $PM_DIR $PT_DIR $REV_DIR $BDY_DIR

# the following command should be run separately

####################################
##### prepare data for stage 1 #####
####################################


# 3. combine the aig, syn_aig, sub_aig and pm as a .pt file
python ./src/data_process/3_gen_pkl_for_subgraph_mining.py --aig_dir $AIG_DIR --pm_dir $PM_DIR --save_path $PT_DIR --num_workers $NUM_WORKERS;

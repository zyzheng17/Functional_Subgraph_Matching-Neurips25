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

# 3. merge the data and split train and test
python ./src/data_process/6_merge_pkl.py --input_folder $BDY_DIR --output_file $BDY_DIR;

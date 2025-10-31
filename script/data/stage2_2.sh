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

# 2. generate aig from pm netlist
python ./src/data_process/5_parse_pm_aig_verilog.py --pm_root $PM_DIR --pm_aig_root $REV_DIR --save_root $BDY_DIR;

# Data Processing
DATASET_NAME=itc99 #itc99 openabcd forgeeda
SAMPLE_RATIO=0.2 ## 0.2 for ITC99; 0.05 for OpenABCD, 0.003 for ForgeEDA
RAW_DATA=./raw_data/$DATASET_NAME
AIG_DIR=./dataset/$DATASET_NAME/aig
PM_DIR=./dataset/$DATASET_NAME/pm
PT_DIR=./dataset/$DATASET_NAME/pm_aig
REV_DIR=./dataset/$DATASET_NAME/pm2aig
BDY_DIR=./dataset/$DATASET_NAME/boundary
NUM_WORKERS=1
ABC_PATH=/home/zyzheng23/projects/abc/abc


mkdir -p $AIG_DIR $PM_DIR $PT_DIR $REV_DIR $BDY_DIR

# the following command could run separately for debug

####################################
##### prepare data for stage 1 #####
####################################

# 1. cut the raw data to generate aig, syn_aig, and sub_aig
python ./src/data_process/1_cut_cone.py --aig_dir $RAW_DATA --save_root $AIG_DIR --abc_path $ABC_PATH  --sample_ratio $SAMPLE_RATIO  --num_workers $NUM_WORKERS;

# 2. generate the pm netlist for aig
python ./src/data_process/2_aig_mapping.py --aig_dir $AIG_DIR --pm_dir $PM_DIR --abc_path $ABC_PATH --num_workers $NUM_WORKERS;

# 3. combine the aig, syn_aig, sub_aig and pm as a .pt file
python ./src/data_process/3_gen_pkl_for_subgraph_mining.py --aig_dir $AIG_DIR --pm_dir $PM_DIR --save_path $PT_DIR --num_workers $NUM_WORKERS;

# 4. merge the data and split train and test
python ./src/data_process/6_merge_pkl.py --input_folder $PT_DIR --output_file $PT_DIR ;

####################################
##### prepare data for stage 2 #####
####################################

# 1. extract aig for the node in pm netlist
python ./src/data_process/4_pm_to_aig.py --data_path $PM_DIR --save_path $REV_DIR;

# 2. generate aig from pm netlist
python ./src/data_process/5_parse_pm_aig_verilog.py --pm_root $PM_DIR --pm_aig_root $REV_DIR --save_root $BDY_DIR;

# 3. merge the data and split train and test
python ./src/data_process/6_merge_pkl.py --input_folder $BDY_DIR --output_file $BDY_DIR;

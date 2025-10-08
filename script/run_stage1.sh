MODEL_LIST=DeepGate2,NeuroMatch,ABGNN,Gamora,HGCN
MODEL=DeepGate2 
TRAIN_DATA=./dataset/itc99/itc99_pm_aig_train.pt
TEST_DATA=./dataset/itc99/itc99_pm_aig_test.pt
LOG_PATH=./stage1_log

python ./src/train.py --encoder_type $MODEL --train_data $TRAIN_DATA --test_data $TEST_DATA --log_path $LOG_PATH
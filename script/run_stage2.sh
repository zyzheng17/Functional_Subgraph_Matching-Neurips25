MODEL_LIST=DeepGate2,NeuroMatch,ABGNN,Gamora,HGCN
MODEL=DeepGate2 
TRAIN_DATA=./dataset/itc99/itc99_boundary_identify_train.pt
TEST_DATA=./dataset/itc99/itc99_boundary_identify_test.pt
LOG_PATH=./stage2_log
PRETRAIN_PATH=/path/to/pretrained/model.ckpt

python ./src/train_boundary.py --encoder_type $MODEL --train_data $TRAIN_DATA --test_data $TEST_DATA --log_path $LOG_PATH --pretrain_path $PRETRAIN_PATH
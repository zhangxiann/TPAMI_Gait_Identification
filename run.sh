# cnn

## genetate img
python make_hdf5.py

nohup python -u train_gait_cnn.py --img_type time_only_two_channel --epoch 50 --cuda 0 --batch_size 128 > time_only_two_channel.log 2>&1 &
nohup python -u train_gait_cnn.py --img_type counts_only_two_channel --epoch 50 --cuda 1 --batch_size 128 > counts_only_two_channel.log 2>&1 &
nohup python -u train_gait_cnn.py --img_type counts_and_time_two_channel --epoch 50 --cuda 0 --batch_size 128 > counts_and_time_two_channel.log 2>&1 &
nohup python -u train_gait_cnn.py --img_type four_channel --epoch 50 --cuda 1 --batch_size 128 > four_channel.log 2>&1 &

python test_gait_cnn.py --img_type four_channel --model_name EV_Gait_IMG_four_channel.pkl
python test_gait_cnn.py --img_type counts_only_two_channel --model_name EV_Gait_IMG_counts_only_two_channel.pkl
python test_gait_cnn.py --img_type counts_and_time_two_channel --model_name EV_Gait_IMG_counts_and_time_two_channel.pkl
python test_gait_cnn.py --img_type time_only_two_channel --model_name EV_Gait_IMG_time_only_two_channel.pkl

# gcn


nohup python -u EV-Gait-3DGraph/train_3d_graph.py --epoch 150 --cuda 0 > train_3d_graph.log 2>&1 &



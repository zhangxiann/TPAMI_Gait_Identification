nohup python -u train_gait_cnn.py --img_type time_only_two_channel --epoch 100 --cuda 0 > time_only_two_channel.log 2>&1 &
nohup python -u train_gait_cnn.py --img_type counts_only_two_channel --epoch 100 --cuda 0 > counts_only_two_channel.log 2>&1 &
nohup python -u train_gait_cnn.py --img_type counts_and_time_two_channel --epoch 100 --cuda 0 > counts_and_time_two_channel.log 2>&1 &
nohup python -u train_gait_cnn.py --img_type four_channel --epoch 100 --cuda 0 > four_channel.log 2>&1 &

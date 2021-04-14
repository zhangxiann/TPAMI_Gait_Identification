# cnn
nohup python -u train_gait_cnn.py --img_type time_only_two_channel --epoch 60 --cuda 0 > time_only_two_channel.log 2>&1 &
nohup python -u train_gait_cnn.py --img_type counts_only_two_channel --epoch 80 --cuda 1 > counts_only_two_channel.log 2>&1 &
nohup python -u train_gait_cnn.py --img_type counts_and_time_two_channel --epoch 60 --cuda 0 > counts_and_time_two_channel.log 2>&1 &
nohup python -u train_gait_cnn.py --img_type four_channel --epoch 60 --cuda 1 > four_channel.log 2>&1 &

# gcn
nohup python -u train_3d_graph.py --epoch 150 --cuda 1 > train_3d_graph.log 2>&1 &

nohup python -u train_3d_graph.py --epoch 150 --cuda 1 > train_3d_graph_no_polarity.log 2>&1 &



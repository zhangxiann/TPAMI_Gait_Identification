function pcd_downsampler(data_path, save_path, max_points)

	[time, x, y, p] = textread(data_path, '%f %f %f %f');
	% normalize time to start from 0
	% get the first timestamp
	start = time(1);
	time = time - start;
	data = [time, x, y, p];
	% some data has less event, these data cannot generate edges, so drop them
	%size(x)
	%if (size(x)) > 1000
	points = downsample(data, max_points);
	save([save_path, '.mat'], 'points');
	save_path
	% end
end

function [down_data] = downsample(data, max_points)
	time = data(:, 1);
	x = data(:, 2);
	y = data(:, 3);
	
	p = data(:, 4);
	% get the last timestamp
	time_length = time(size(time, 1));
	%rescale the timestampes to start from 0 up to 128
	time = time / time_length * 128;
	%convert to point cloud object
	points = [x, y, time];
	ptCloud = pointCloud(points);
	% down sample by using the functions from MATLAB
	ptCloudOut = pcdownsample(ptCloud, 'nonuniformGridSample', max_points);
	%extract the result from the original data
	% points_downsample: [x, y, time]
	points_downsample = ptCloudOut.Location;
	[~, pos] = ismember(points_downsample, points, 'rows');
	% need to add polarity, so get data from original points
	down_data = [];
	% iterate pos(index)
	for i = pos
		% add event to down_data
		down_data = [down_data; x(i), y(i), time(i), p(i)];
		%     down_data = [down_data; time(i), x(i), y(i), p(i)];
	end
end
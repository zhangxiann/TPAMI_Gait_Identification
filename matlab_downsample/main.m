clc;
close all;
clear;

% hyper parameter MaxNumEvents
max_points = 40;
res_root = '..\data\DVS128-Gait-Day\origin';
des_root = '..\data\DVS128-Gait-Day\downsample';

% delete old downsample directory
if exist(des_root,'dir')==1
    rmdir(des_root, 's');
end    

train_test = dir(res_root);
scene_num = int8(size(train_test, 1)-2);

parfor i=1:scene_num
	scene=train_test(i+2).name;
	people = dir([res_root filesep scene]);
    label = -1;
    for p=1:length(people)

        if( isequal( people( p ).name, '.' )||...
            isequal( people( p ).name, '..')||...
            ~people( p ).isdir)               % skip other file
            label = -1;
            continue;
        end
        label = str2num(people(p).name);
	    if ~exist([des_root filesep scene filesep num2str(label)], 'dir')
                mkdir([des_root, filesep, scene, filesep, num2str(label)]);
        end
        

        count = dir([res_root filesep scene filesep people( p ).name]);
        for j = 1:length(count)
            if( isequal( count(j).name, '.' )||...
                isequal( count(j).name, '..')||...
                count(j).isdir)                % skip other file
                continue;
            end

            data_path = [res_root, filesep, scene, filesep, people(p).name, filesep, count(j).name];
            save_path = [des_root, filesep, scene, filesep, num2str(label), filesep, num2str(j)];
            pcd_downsampler(data_path, save_path, max_points);
			% scene+" "+subject+" "+j
        end
    end

end
fprintf("Finshed\n");

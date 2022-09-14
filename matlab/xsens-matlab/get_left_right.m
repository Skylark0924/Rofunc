T8_data_pos = segmentData(5).position;
T8_data_ori = segmentData(5).orientation;
T8_data = [T8_data_pos, T8_data_ori];

right_tcp_data_pos = segmentData(11).position;
right_tcp_data_ori = segmentData(11).orientation;
right_tcp_data = [right_tcp_data_pos, right_tcp_data_ori];


left_tcp_data_pos = segmentData(15).position;
left_tcp_data_ori = segmentData(15).orientation;
left_tcp_data = [left_tcp_data_pos, left_tcp_data_ori];

mkdir Matlab_files/HIKE_Dataset/csv_data box_carrying_001
name = 'box_carrying_001';
writematrix(T8_data, sprintf('Matlab_files/HIKE_Dataset/csv_data/%s/T8.csv', name))
writematrix(right_tcp_data, sprintf('Matlab_files/HIKE_Dataset/csv_data/%s/right.csv', name))
writematrix(left_tcp_data, sprintf('Matlab_files/HIKE_Dataset/csv_data/%s/left.csv', name))

% save('T8.mat', 'T8_data')
% save('right.mat', 'right_tcp_data')
% save('left.mat', 'left_tcp_data')
clear
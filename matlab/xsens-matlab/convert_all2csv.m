json_data = jsonencode(tree);

path = 'Matlab/Dough_Dataset/06_24';
mkdir (path);
name = 'dough_01';
% writematrix(json_data, sprintf("%s/%s.csv", path, name));
saveJSONfile(tree.fingerDataLeft, sprintf("%s/%s.json", path, name));
clear
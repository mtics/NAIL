% Version: 1.0, created on 08/03/2021, modified on 08/04/2021,
% Author: Zhiwei Li

clear; clc;

%% Set the datesets you want to transfer here
vecSets = {'corel5k', 'espgame', 'mirflickr', 'pascal07', 'iaprtc12'}; % their fomates are ".*vec*"

arffSets = {'emotions', 'yeast'}; % their fomates are ".arff"

partitions = {'test', 'train'}; % Used to distinct train sets and test sets.

%% If you want to run this file on your own PC,
%% These two paths must be modified to your own locations.
loadPath = '../datasets/datasets\';
savePath = 'datasets\';

emotions_views = {'Timbre', 'Thythmic', 'Label'};
emotions_splits = [64, 8, 6];
yeast_views = {'Genetic', 'Phylogenetic', 'Label'};
yeast_splits =  [79, 24, 14];

% This loop is used to read vecSets, since their file is ".*vec(s)"
for set = vecSets
    
    for partition = partitions
        vec2mat(loadPath, set, partition, 'DenseSift', 'hvecs', savePath);
        vec2mat(loadPath, set, partition, 'DenseHue', 'hvecs', savePath);
        vec2mat(loadPath, set, partition, 'Gist', 'fvec', savePath);
        vec2mat(loadPath, set, partition, 'Hsv', 'hvecs32', savePath);
        vec2mat(loadPath, set, partition, 'Rgb', 'hvecs32', savePath);
        vec2mat(loadPath, set, partition, 'Lab', 'hvecs32', savePath);
        vec2mat(loadPath, set, partition, 'annot', 'hvecs', savePath);
    end
    
    data_merge(savePath, set, 'DenseSift', savePath);
    data_merge(savePath, set, 'DenseHue', savePath);
    data_merge(savePath, set, 'Gist', savePath);
    data_merge(savePath, set, 'Hsv', savePath);
    data_merge(savePath, set, 'Rgb', savePath);
    data_merge(savePath, set, 'Lab', savePath);
    data_merge(savePath, set, 'Label', savePath);
    
    fprintf(['Transferring the format of ', char(set), ' is finished.\n']);
    
end

% This loop is used to read arffSets, since their file is ".arff"
for set = arffSets
    
    dataPath = [loadPath, char(set), '/'];
    
    files = dir([dataPath, '*.arff']);
    
    for i = 1:length(files)
        views = eval([char(set), '_views']);
        splits = eval([char(set), '_splits']);
        arff2mat(dataPath, char(set), files(i).name, views, splits, savePath);
    end
    
    fprintf(['Transferring the format of ', char(set), ' is finished.\n']);
    
end

fprintf('All datasets are transferred.\n');


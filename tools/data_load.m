function[X, Y] = data_load(dataPath, selectedSet, partition)
% DATALOAD load all data from the selected dataset,
% and return the feature matrices X, and the label matrix Y.
%
% Input:
%   dataPath: cell array, contains features matrices or a label matrix
%   selectedSet: the selected dataset
%   partition: 'test' or 'train'
%
% Output:
%   X: cell, contains all feature matrices
%   Y: cell, contains the label matrix
%
% Call:
%   [X, Y] = data_load(dataPath, selectedSet, partition)
%
% Version: 1.0, created on 08/04/2021, modified on 03/14/2022,
% Author: Zhiwei Li

% load files
datasetPath = strcat(dataPath, selectedSet, '/');

if (nargin < 3)
    % This will load total dataset (no partition).
    files = [];
    % I didn't understand how to match files have no 'test' or 'train' in their names.
    tempFiles = dir(strcat(datasetPath, selectedSet, '_*.mat')); 
    for i = 1:length(tempFiles)
        if isempty(strfind(tempFiles(i).name, 'test')) && isempty(strfind(tempFiles(i).name, 'train'))
            files = [files tempFiles(i)];
        end
    end
else
    % This will load datasets according to the partition
    files = dir(strcat(datasetPath, '*_', partition, '_*.mat'));
end

% Get the number of views
views = length(files)-1;

X = cell(1, views);
Y = cell(1, 1);

index = 1; % used to index views
for i = 1:length(files)
    % Only load feature matrices here
    if isempty(strfind(files(i).name, 'Label'))
        X{index} = double(cell2mat(struct2cell(load(strcat(datasetPath, files(i).name))))); % struct -> cell -> mat
        X{index} = normalize(X{index}, 1); % Data Normalization
        X{index}(isnan(X{index})) = 0;
        index = index+1;
    else
        % Load the label matrix here
        Y{1} = double(cell2mat(struct2cell(load(strcat(datasetPath, files(i).name))))); % struct -> cell -> mat
    end
end
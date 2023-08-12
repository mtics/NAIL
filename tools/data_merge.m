function [] = data_merge(loadPath, selectedSet, fileName, savePath)
% DATA_MERGE merge *_test_*.mat and *_train_*.mat into one mat.
%
% Input:
%   loadPath: The path where you store the raw dataset
%   selectedSet: Specify which dataset is transferred
%   fileName: Specify which file is transferred 
%
% Output:
%   none
%
% Call:
%   data_merge(loadPath, selectedSet, fileName, savePath)
%
% Version: 1.0, created on 08/13/2021, modified on 08/13/2021,
% Author: Zhiwei Li

% cell --> char
selectedSet = char(selectedSet);
datasetPath = [loadPath, selectedSet, '/'];
name = [datasetPath, selectedSet, '_*_', fileName, '.mat'];

files = dir(name);

savedData = cell2mat(struct2cell(load([datasetPath, files(1).name])));
for i = 2:length(files)
    savedData = [savedData; cell2mat(struct2cell(load([datasetPath, files(i).name])))];
end

% Save data in '*.mat'
savedFile = [savePath, selectedSet, '/', selectedSet, '_', fileName, '.mat'];
save(savedFile, 'savedData');

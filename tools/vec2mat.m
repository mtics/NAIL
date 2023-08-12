function[] = vec2mat(loadPath, selectedSet, partition, fileName, format, savePath)
% VEC2MAT transfer '.*vec*' to '.mat'. 
% Now supported extensions:
% 'fvec(s)', 'dvec(s)', 'hvec(s)', 'hvec(s)32'
%
% Input:
%   loadPath: The path where you store the raw dataset
%   selectedSet: Specify which dataset is transferred
%   partition: 'train' or 'test'
%   fileName: Specify which file is transferred 
%   savePath: The path where you want to store the transferred dataset.
%
% Output:
%   none
%
% Version: 1.0, created on 08/03/2021, modified on 08/04/2021,
% Author: Zhiwei Li


% cell --> char
selectedSet = char(selectedSet);
partition = char(partition);

dataFile = [loadPath, selectedSet, '/', selectedSet, '_', partition, '_', fileName, '.', format];

% If "*_annot.*" is not existed, use "*_classes.txt" instead
if strcmp(fileName, 'annot') && ~exist(dataFile, 'file')
    fileName = 'classes';
    format = 'txt';
    dataFile = [loadPath, selectedSet, '/', selectedSet, '_', partition, '_', fileName, '.', format];
end

% The label data in differen sets have different names,
% so we use a standard name 'Label' instead.
if strcmp(fileName, 'annot') || strcmp(fileName, 'classes')
    savedName = 'Label';
else
    savedName = fileName;
end

savedData = [selectedSet, '_', partition, '_', savedName];

if exist(dataFile, 'file')
    
    % Load data
    if strcmp(format, 'txt')
        loadCommand = [savedData, ' = load("', dataFile, '");'];
    else
        loadCommand = [savedData, ' = vec_read("', dataFile, '");'];
    end
    
    eval(loadCommand);
    
    folder = [savePath, selectedSet, '/'];
    % If the folder used is not existed, create it first
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    
else
    % In some datasets we used, there is '*_tags.txt' instead of '*_annot(or tags).hvecs',
    % so it will use another way to read the file.
    format = 'txt';
    dataFile = [loadPath, selectedSet, '/', selectedSet, '_', partition, '_', fileName, '.', format];
    eval([savedData, '= dlmread("', dataFile, '");']);
end

% Save data in '*.mat'
savedFile = [savePath, selectedSet, '/', selectedSet, '_', partition, '_', savedName, '.mat'];
save(savedFile, savedData);

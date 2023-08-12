function[] = arff2mat(dataPath, selectedSet, fileName, views, splits, savePath)
% ARFF2MAT transfer '.arff' to '.mat', and can split a matrix with multiple
% views to many matrices, each one corresponds to one view.
%
% Input:
%   dataPath: The path where you store the raw dataset
%   selectedSet: Specify which dataset is transferred
%   fileName: Specify which file is transferred 
%   views: A set specifies the name of each view, which length should be
%          equal to the set 'splits'.
%   splits: A set specifies the dimensions of each view, which length should be
%           equal to the set 'splits'.
%   savePath: The path where you want to store the transferred dataset.
%
% Output:
%   none
%
% Version: 1.0, created on 08/03/2021, modified on 08/03/2021,
% Author: Zhiwei Li

% If the lengths of 'views' and 'splits' are not equal,
% there must be some error.
if length(views) ~= length(splits)
   fprintf('The lengths of views and splits are not same!'); 
   return;
end

% cell -> char
fileName = char(fileName);

filePath = [dataPath, '/', fileName];

% load '.arff' and store data in the 'mat' format
wekaObj = loadARFF(filePath);
[wholeData,~,~,~,~] = weka2matlab(wekaObj, []);
splitedData = mat2cell(wholeData, size(wholeData, 1), splits);

folder = [savePath, selectedSet, '/'];
% If the folder used is not existed, create it first
if ~exist(folder, 'dir')
    mkdir(folder);
end

% Data of different views would be stored in different matrices. 
for i = 1:length(views)
    
    [strs, ~] = strsplit(fileName, '.');
    
    savedData = [char(strrep(strs{1}, '-', '_')), '_', char(views{i})];
    eval([savedData, ' = zeros(size(splitedData{i}));']);
    eval([savedData, ' = splitedData{i};']);
    
    savedFile = [savePath, selectedSet, '/', savedData, '.mat'];
    save(savedFile, savedData);
end
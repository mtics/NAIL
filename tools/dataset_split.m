function[trainX, trainY, testX, testY, trainRows, testRows] = dataset_split(X, Y, trainRatio)
% DATASET_SPLIT used to splite a whole dataset into two part:
%     a train set and a test set.
%
% Input:
%   X: cell array, contains all feature matrices
%   Y: cell array, contains the label matrix
%
% Output:
%   trainX: cell array, the train set of feature matrices
%   trainY: cell array, the train set of label matrix
%   testX: cell array, the test set of feature matrices
%   testY: cell array, the test set of label matrices
%
% Call:
%   [trainX, trainY, testX, testY] = dataset_split(X, Y, trainRatio)

% Version: 1.0, created on 08/05/2021, modified on 03/14/2022,
% Author: Zhiwei Li

% Define local parameters
[samples, labels] = size(Y{1});
views = length(X);

trainX = cell(1, views);
testX = cell(1, views);
trainY = cell(1, 1);
testY = cell(1, 1);

trainNum = floor(samples * trainRatio); % Set the train number

% Ensure there is at least one sample for each label
indexTrainKeep = zeros(1, labels);
for i = 1:labels
   indexPositive = find(Y{1}(:, i) == 1);
   indexTrainKeep(i) = indexPositive(1); 
end

indexTrainKeep = unique(indexTrainKeep);

% Get the row indexes of the train set and test set
randIndex = randperm(samples);
randIndex = setdiff(randIndex, indexTrainKeep);

trainRows = [randIndex(1:trainNum-length(indexTrainKeep)), indexTrainKeep];
testRows = setdiff(randIndex, trainRows);

% Split the original dataset into two subsets
for v = 1:views
    rawX = X{v};
    trainX{v} = rawX(trainRows, :);
    testX{v} = rawX(testRows, :);
end

trainY{1} = Y{1}(trainRows, :);
testY{1} = Y{1}(testRows, :);







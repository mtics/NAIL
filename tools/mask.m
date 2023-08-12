function [maskedData, Observed] = mask(rawData, maskRatio)
% MASK randomly set the elements in rawData to zero at maskRatio.
% It would generate a masked cell array with all masked data to 0,
%
% ** Notice ** that in our experiments, no feature data have only a
% single view, and its' feature dimensions are all not equal to the
% label dimension,
%
% Input:
%   rawData: cell array, contains features matrices or a label matrix
%   maskRatio: double, the mask ratio
%   rngSeed: int, a seed id to contrl the random nuber generator
%
% Output:
%   maskedData: cell array, the matrices after masked
%   Observed: cell array, used to store the index of observed elements
%
% Call:
%   [maskedData, Observed] = mask(rawData, maskRatio, rngSeed)
%
% Version: 1.0, created on 08/04/2021, modified on 08/11/2022,
% Author: Zhiwei Li


%% Note:
%   Since we didn't know which matrix would be passed in,
%   we cannot specify either paras.d or paras.l to use.
%   Thus, we still need to get the sizes.
%%

%% Function implementation
samples = size(rawData{1}, 1); % the number of samples
views = length(rawData);     % the number of views;

deleteIndex = ones(samples, views);
indicator = cell(1, views);
Observed = cell(1, views); % used to save indexs
maskedData = rawData;

if views == 1
    % The outer if-else is used to distinct label matrix and feature
    % matrix.
    Observed{1} = ones(size(rawData{1}));
    labels = size(rawData{1}, 2);
    for i = 1:labels
        rowIndexPositive = find(rawData{1}(:, i) ~= 0);
        unObservedIndex = randperm(length(rowIndexPositive), max(floor(length(rowIndexPositive) * maskRatio), 0));
        Observed{1}(unObservedIndex, i) = 0;
        
        rowIndexNegative = find(rawData{1}(:, i) == 0);
        unObservedIndex = randperm(length(rowIndexNegative), max(floor(length(rowIndexNegative) * maskRatio), 0));
        Observed{1}(unObservedIndex, i) = 0;
    end
    
    % Generate masked data matrices
    maskedData{1} = Observed{1} .* rawData{1};
%     Observed{1} = maskedData{1};
    
else
    

    % We mask feature matrices here.
    for v = 1:views
        if v ~= views
            index = randperm(samples, max(floor(samples * maskRatio), 0));
            deleteIndex(index, v) = 0;
            indicator{v} = index;
        else
            % Keep each sample at least completed on one view.
            keepIndex = find(sum(deleteIndex(:, 1:views-1), 2) == 0);
            
            % Finde indexes that can be deleted.
            mayDeleteIndex = setdiff(1:samples, keepIndex);
            tempIndex = randperm(length(mayDeleteIndex), max(floor(samples * maskRatio), 0));
            
            index = mayDeleteIndex(tempIndex);
            deleteIndex(index, v) = 0;
            indicator{v} = index;
        end
    end
    
    for v = 1:views
        % Generate mask
        Observed{v} = ones(size(maskedData{v}));
        Observed{v}(indicator{v}, :) = 0;
        
        maskedData{v} = maskedData{v} .* Observed{v};
    end
end

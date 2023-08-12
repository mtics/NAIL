function [noiseData] = add_noise(rawData, noise_ratio)
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

noiseData = rawData;

 % We mask feature matrices here.
for v = 1:views
    dimension = size(rawData{v}, 2);
    for i = 1:samples
        noiseIndex = randperm(dimension, max(floor(dimension * noise_ratio), 0));
        noiseData{v}(i, noiseIndex) = gaussian_noise(1, length(noiseIndex), 5, 0.1);
    end
end

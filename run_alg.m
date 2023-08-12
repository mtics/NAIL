function[result] = run_alg(method, kernel, dataset, isFixed, lambda, mu, subRatio, sigma, fmr, lmr)
% run_alg is used to run our method "cement" or "rimar".
% 
% Input:
%   method: 'cement' or 'rimar'
%   kernel: 'linear' or 'gaussian'
%   dataset: 'corel5k', 'espgame', 'mirflickr', 'pascal07', 'iaprtc12', 'yeast', 'emotions'
%   isFixed: true or false, used to set the step sizes automatically or manually.
%   lambda: the balance term
%   subRatio: the ratio of k/d
%   sigma: the value used in gaussian kernel
%   fmr: the feature mask ratio
%   lmr: the label mask ratio
%
% Call:
%   run_alg(method, kernel, dataset, isFixed, lambda, subRatio, sigma, fmr, lmr)

addpath('tools');
addpath('evaluations');
addpath(method);

sigma = 1;

% If the changed param is already in params, it can be changed as
% following:
%       params = initializeProject("lambda", 5, "decay_rate", 12);
% Else, it should be added into params:
%       params.a = 1;
params = initializeProject( ...
    'lambda', lambda, ...
    'mu', mu, ...
    'subRatio', subRatio, ...
    'sigma', sigma, ...
    'featureMaskRatio', fmr, ...
    'labelMaskRatio', lmr, ...
    'kernel', kernel, ...
    'isFixed', isFixed, ...
    'device', 'cpu' ...
);

params.set = dataset;

%% Load data
sampleRatio = 1;
switch params.set
    case 'corel5k'
        sampleRatio = 0.2;
    case 'espgame'
        sampleRatio = 0.05;
    case 'mirflickr'
        sampleRatio = 0.05;
    case 'pascal07'
        sampleRatio = 0.1;
    case 'iaprtc12'
        sampleRatio = 0.05;
    case 'yeast'
        sampleRatio = 1;
    case 'emotions'
        sampleRatio = 1;
    case 'simulation'
        sampleRatio = 1;
    otherwise
        sampleRatio = 0.1;
end

trainset = struct();

rng('default');

%%% Real Dataset
[realX, realY] = data_load('datasets/', params.set);

% Only use 20/50 labels
% [label_nums, idx] = sort(sum(realY{1}), 'descend');
% realY{1} = realY{1}(:, idx(1:min(50, length(idx))));

[trainset.X, trainset.Y, ~, ~] = dataset_split(realX, realY, sampleRatio);    % Only use part of large sets to save time

views = length(trainset.X); % The number of views

% Decrease the feature dimensions to save time
for v = 1:views
    [~, score, ~] = pca(trainset.X{v});
    feature = min(size(score, 2), 50);
    trainset.X{v} = score(:, 1:feature);
end

%% Train
fprintf('The mask ratios: [r: %.1f] [s: %.1f].\n', params.featureMaskRatio, params.labelMaskRatio);
fprintf('Training %s on the %s dataset starts at %s.\n', method, params.set, datestr(now, 'yyyy-mm-dd HH:MM:SS'));

evaluations = zeros(params.round, 5);
result = zeros(1, 14);

% run params.round times
for iter = 1:params.round

    fprintf('Training %s on the %s dataset at %d-th iter\n', method, params.set, iter);

    params.seed = rng(iter);
    
    % split the dataset
    [trainset.X, trainset.Y, trainset.tX, trainset.tY, trainset.trainRows, trainset.testRows] = dataset_split(trainset.X, trainset.Y, 0.8);    % Spit the dataset
    
    for v = 1:views
        trainset.X{v} = [trainset.X{v}; trainset.tX{v}];
    end

    % Generate masked data and the corresponding observed indexes
    [trainset.maskedX, trainset.Ox] = mask(trainset.X, params.featureMaskRatio);
    [trainset.maskedY, trainset.Oy] = mask(trainset.Y, params.labelMaskRatio);

    [evaluations(iter, :), outputs] = call_rimar(trainset, params);


end

fprintf('The mask ratios: [r: %.1f] [s: %.1f].\n', params.featureMaskRatio, params.labelMaskRatio);
fprintf('Training %s on the %s dataset ends at %s.\n', method, params.set, datestr(now, 'yyyy-mm-dd HH:MM:SS'));

result(1:4) = [lambda, mu, subRatio, sigma]; 

% Average evaluations
evaluations((all(evaluations==0,2)), :) = [];
result(5:9) = mean(evaluations, 1);
result(10:14) = std(evaluations, 0, 1);

saveFolder = strcat('FM-', num2str(params.featureMaskRatio), '-LM-', num2str(params.labelMaskRatio));
savePath = get_folder(strcat('outputs/', method, '/', params.set, '/', params.kernel, '/', saveFolder));

saveName = strcat('evaluation_', num2str(lambda), '_', num2str(mu), '_', num2str(subRatio), '_', num2str(sigma));
parsave(savePath, saveName, '.mat', result);

saveName = strcat('trainset_', num2str(lambda), '_', num2str(mu), '_',  num2str(subRatio), '_', num2str(sigma));
parsave(savePath, saveName, '.mat', outputs);

rmpath(method);

end
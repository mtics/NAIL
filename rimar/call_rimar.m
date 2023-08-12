function[evaluation, trainset] = call_rimar(trainset, params)
% CALL_CEMENT used to call the RIMAR method.
%
% Input:
%   - trainset: struct, stored all train data, including:
%           - Ox: cell array, the obeserved feature indicator.
%           - Oy: cell array, the obeserved label indicator
%           - X: cell array, the feature matrices
%           - Y: cell array, the label matrix
%           - maskedX: cell array, the masked feature matrices
%           - maskedY: cell array, the masked label matrix
%
%   - params: struct, stored some pre-defined scalar parameters, including:
%           - lambda: the balance term
%           - tol: the tolerance
%           - featureMaskRatio: the mask ratio of features
%           - labelMaskRatio: the mask ratio of labels
%           - maxIter: the max iteration times
%           - round: the run times
%           - decay_rate: decrease the step size of gradient descends
%           - step_size: the step size of gradient descends
%           - subRatio: the ratio of subspaces to origin spaces, i.e. k(v)/d(v)
%           - set: the selected dataset
%
% Output:
%   evaluations: array, used to store four evaluations, in order:
%           1) 1-HL    2) 1-RL    3) AP    4) AUC
%   trainset: the learned matrices
%
% Call:
%   [evaluations] = call_cement(trainset, params)
%   [evaluations, trainset] = call_cement(trainset, params)
%
% Version: 1.0, created on 08/29/2021, modified on 03/23/2022,
% Author: Zhiwei Li

%% Add path
addpath("rimar/model_tools");

% switch params.set
%     case 'corel5k'
%         params.lambda = 1e-2;
%         params.subRatio = 0.2;
%         if strcmp(params.kernel, 'gaussian')
%             params.subRatio = 0.5;
%         end
%     case 'espgame'
%         params.lambda = 1e-2;
%         params.subRatio = 0.2;
%         if strcmp(params.kernel, 'gaussian')
%         end
%     case 'mirflickr'
%         params.lambda = 1e-2;
%         params.subRatio = 0.2;
%         if strcmp(params.kernel, 'gaussian')
%             params.subRatio = 0.5;
%         end
%     case 'pascal07'
%         params.lambda = 1e-2;
%         params.subRatio = 0.5;
%         if strcmp(params.kernel, 'gaussian')
%             params.subRatio = 0.2;
%         end
%     case 'iaprtc12'
%         params.lambda = 1e-2;
%         params.subRatio = 0.2;
%         if strcmp(params.kernel, 'gaussian')
%             params.subRatio = 0.5;
%         end
%     case 'yeast'
%         params.lambda = 1e-5;
%         params.subRatio = 0.2;
%         if strcmp(params.kernel, 'gaussian')
%         end
%     case 'emotions'
%         params.lambda = 1e-1;
%         params.subRatio = 0.8;
%         if strcmp(params.kernel, 'gaussian')
%             params.lambda = 1e-3;
%             params.subRatio = 0.2;
%         end
%     case 'simulation'
%         params.maxIter = 3000;
%         params.lambda = 2.5e-1;
%         params.subRatio = 0.5;
%     otherwise
%         params.lambda = 1e-2;
%         params.subRatio = 0.5;
% end

% Define local parameters
trainset.Oy = cell2mat(trainset.Oy);

m = length(trainset.X); % the number of views

% add some noise
rng("default");

[r, c] = size(trainset.X{1});
trainset.X{1} = gaussian_noise(r, c, 0.0128, sqrt(0.9596));

[r, c] = size(trainset.X{2});
trainset.X{2} = trainset.X{2} + gaussian_noise(r, c, 0.0128, sqrt(0.9596));

% Dataset preprocess
needMap = false;
for v = 1:m
    if min(min(trainset.X{v}) < 0)
        needMap = true;
        break;
    end
end

if needMap
    for v = 1:m
        trainset.X{v} = mapminmax(trainset.X{v}, 0, 1);
    end
end

trainset.Y{1}(trainset.Y{1} <= 0) = 0;
trainset.tY{1}(trainset.tY{1} <= 0) = 0;

maskedX = cell(1,m);
for v = 1:m
    maskedX{v} = trainset.Ox{v} .* trainset.X{v};
end

Y = [trainset.Y{1}; trainset.tY{1}];
Oy = [trainset.Oy; zeros(size(trainset.tY{1}))];
% Oy = ones(size(Y));
maskedY = Oy.*Y;

params.trainNum = length(trainset.trainRows);
params.testNum = length(trainset.testRows);

%% Train

% baseline(trainset.Ox, Oy, trainset.X, Y, params);

outputs = rimar(trainset.Ox, Oy, trainset.X, Y, params);
if outputs.convergencedIter == -1
    outputs.convergencedIter = params.maxIter;
end

%% Evaluations
evaluation = zeros(1, 5);
evaluation(1:4) = evaluate(outputs.hatY(params.trainNum+1:size(Y, 1), :), trainset.tY{1});
for v = 1:m
    evaluation(5) = evaluation(5) + norm((1-trainset.Ox{v}).*(outputs.hatX{v}-trainset.X{v}), "fro") / norm(((1-trainset.Ox{v}) .* trainset.X{v}), "fro");
end
evaluation(5) = evaluation(5) / m;

evaltrain = evaluate(outputs.hatY(1:params.trainNum, :), trainset.Y{1});

%% Plot

% figure(1);
% subplot(2,2,1);
% plot(outputs.losses(1, 1:outputs.convergencedIter));
% subplot(2,2,2);
% plot(outputs.losses(2, 1:outputs.convergencedIter));
% subplot(2,2,3);
% plot(outputs.losses(3, 1:outputs.convergencedIter));
% subplot(2,2,4);
% plot(outputs.losses(4, 1:outputs.convergencedIter));

trainset.outputs = outputs;

% if strcmp(params.set, 'simulation')
%     addpath("plots/");
%     plot_simulation(trainset, maskedX, maskedY, evaluations, losses, convergencedIter);
%     rmpath("plots/");
% end

% if nargout > 1
%     addpath("plots/");
%     plot_simulation(trainset, maskedX, maskedY, evaluations, losses, convergencedIter);
%     rmpath("plots/");
% end

function[evaluation] = evaluate(prediction, truth)
% -----------------------------------------------
%           Evaluate the predicted results      |
% -----------------------------------------------
%
% Input:
%   prediction: the completed Y, not a 0/1 matrix
%   truth: the ground truth Y, a 0/1 matrix
%
% Output:
%   evaluation: contains 4 metrics:
%           1) 1-HL  2) 1-RL  3) AP  4) AUC
%
% Call:
%   [evaluation] = evaluate(prediction, truth)

% Delete the samples don't have labels
index = find(sum(truth, 2) == 0);
truth(index, :) = [];
prediction(index, :) = [];

%% Evaluations
topHatK = ceil(sum(sum(truth))/size(truth, 1)); % Set the hyperparameter k
predLabels = Top_K_Partition(prediction', topHatK)'; % Get the top k elements

% Map negative labels to -1
gtLabels = (truth > 0) - (truth <= 0);

evaluation = zeros(1, 4);
evaluation(1) = 1 - hamming_loss(predLabels', gtLabels');       % Hamming score
evaluation(2) = 1 - ranking_loss(prediction', gtLabels');       % Ranking score
evaluation(3) = average_precision(prediction', gtLabels');      % Average precision
evaluation(4) = average_auroc(prediction', gtLabels');          % AUC

end
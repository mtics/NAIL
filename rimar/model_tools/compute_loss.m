function[loss] = compute_loss(O, X, F, U, alphas, betas, indexes, params)
% COMPUTE_MODEL_LOSS used to compute our model CEMENT's loss
%
% Input:
%   Ox: cell array, the indicator matrices
%   Oy: mat, the indicator matrix
%   X: cell array, feature matrices
%   Y: matrix, label matrix
%   F: cell array, subspaces
%   U: cell array, feature mapping matrices
%   W: matrix, label mapping matrix
%   P: matrix, label probility matrix
%   alphas: array, distances
%   betas: array, importances of different views
%   params: struct, stored some pre-defined scalar parameters, including:
%           - lambda: the balance term
%           - kernel:
%           - sigma:
%
% Output:
%   loss: double, the computed loss of CEMMENT
%
% Call:
%   [loss] = compute_loss(Ox, Oy, X, Y, F, U, W, P, alphas, betas, params)
%
% Version: 1.0, created on 08/06/2021, modified on 10/31/2022,
% Author: Zhiwei Li

m = length(X);

loss = zeros(4, 1);

for v = 1:m
    %% First loss
    dist = max(diag(pdist2(diag(O(v, :))*X{v}, diag(O(v, :))*(F{m+1}+F{v})*U{v})), eps);
    loss(1) = loss(1) + alphas(v)^params.s * sum(dist);
end

%% Second loss
loss(2) = params.lambda * focal_loss(F, U{m+1}, indexes, params);

%% Third Loss
loss(3) = focal_loss(F{m+1}, U{m+1}, indexes, params);

%% Total loss
loss(4) = sum(loss(1:3));

function [X, Y, Pstar, U, W, E] = data_simulation(samples, features, labels)
% DATA_SIMULATION used to generate data that meets our assumptions
%
% Input:
%   samples: int, the number of samples
%   features: array, v-th value is the number of v-th view's features
%   labels: int, the number of labels
%
% Output:
%   X: cell array, the feature matrices
%   Y: cell array, the label matrix
%   W: cell array, label mapping matrices
%   betas: array, importances of different views
%
% Call:
%   [X, Y, W, Yt, S, betas] = data_simulation(samples, features, labels)

% Version: 1.0, created on 08/13/2021, modified on 09/21/2021,
% Author: Zhiwei Li

addpath('cement');

% Define global parameters
global paras;

subRatio = 0.8;
factorP = 1.5;
factorT = 20.5;

m = length(features);
if m ~= 3 && m ~= 4
   fprintf("Simulation setting is error!");
   return;
end

ks = zeros(1, m+1);
for v = 1:m
   ks(v) = floor(subRatio * features(v)); 
end
ks(m+1) = floor(subRatio * labels); 

Pstar = factorP * rand(samples, ks(m+1));

% T = chi2rnd(5, samples, ks(3));
T = factorT * rand(samples, ks(3));

X = cell(1, m); % feature matrices
P = cell(1, m+1);
U = cell(1, m); % label mapping matrices, would be low-rank
for v = 1:m
    if v == 1
        P{v} = Pstar;
       
        % Relative & Low rank     
        U{v} = ones(ks(v), features(v));
        X{v} = P{v} * U{v};    
    elseif v == 2
        P{v} = Pstar;
        
        % Relative & High rank
        X{v} = P{v} * ones(ks(v), features(v)) + factorP * rand(samples, features(v));
    elseif v == 3
        % Irrelative & Low rank
%         U{v} = rand(ks(v), features(v));
        U{v} = ones(ks(v), features(v));
        X{v} = T * U{v};
%         X{v} = T * ones(ks(v), features(v));
%         X{v} = ones(samples, features(v));
    elseif v == 4
%         X{v} = factorT * rand(samples, features(v));
%         X{v} = T * (factorT * U{v}) + factorT * rand(samples, features(v));
        X{v} = T * ones(ks(v), features(v)) + factorT * rand(samples, features(v));
    end
end

% Generate the simulation data
W = rand(ks(m+1), labels);
E = factorP * sprand(samples, labels, 0.3);
% probY =  Pstar * W;
probY =  Pstar * W + E + rand(samples, labels);
for i = 1:labels
    Y{1}(:, i) = (probY(:, i) > mean(probY(:, i)));
end
% Y{1} = mapminmax(probY, 0, 1);
% Y{1} = probY;

paras.samples = samples;
paras.d = features; % d(v) means the feature number of the v-th view
paras.l = labels; % the number of labels
paras.m = m;

function[K] = gaussian_kernel(P, sigma)
% GAUSSIAN_KERNEL used to construct a gaussian kernel of the matrix M,
%
% Input:
%   P: matrix, n x m
%   sigma: scalar
%
% Output:
%   K: matrix, n x n. The gaussian kernel of M.
%
% Call:
%   [K] = gaussian_kernel(P)
%   [K] = gaussian_kernel(P, sigma)
%
% Version: 1.0, created on 03/15/2022, modified on 03/15/2022,
% Author: Zhiwei Li

if nargin == 1
    sigma = mean(mean(P));
end

G = P*P';

g = diag(G);

D = g * ones(n, 1)' + ones(n, 1) * g' - 2 * G;

K = exp(- D / (2 * sigma^2));

end
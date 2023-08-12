function[K] = construct_kernel(P, kernel, sigma)
% GAUSSIAN_KERNEL used to construct a gaussian kernel of the matrix M,
%
% Input:
%   P: matrix, n x m
%   kernel: string, "linear" or "gaussian"
%   sigma: double, used in constructing the gaussian kernel
%
% Output:
%   K: matrix, n x n. The gaussian kernel of M.
%
% Call:
%   [K] = construct_kernel(P)
%   [K] = construct_kernel(P, sigma)
%
% Version: 1.0, created on 03/22/2022, modified on 03/23/2022,
% Author: Zhiwei Li

% Set default values
if nargin == 1
    kernel = 'linear';
    sigma = sqrt(var(var(P)));
elseif nargin == 2
    sigma = sqrt(var(var(P)));
end

% If the value of 'kernel' is not supported,
% print warning infos and finish the procedure.
if ~strcmp(kernel, 'linear') && ~strcmp(kernel, 'gaussian') 
    info = 'Wrong kernel! For now, RIMAR only supports linear or gaussian kernel!';
    disp(info);
    quit;
end

% "sigma == 0" means that the user didn't input the value of 'sigma'.
% Thus, we set 'sigma' to the mean value of the matrix P.
if sigma == 0
    sigma = sqrt(var(var(P)));
end

n = size(P, 1);

if strcmp(kernel, 'linear')
    K = P*P';
elseif strcmp(kernel, 'gaussian')
    G = P*P';

    g = diag(G);

    D = g * ones(n, 1)' + ones(n, 1) * g' - 2 * G;

    K = exp(- D / (2 * sigma^2));
end

end
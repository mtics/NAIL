function[gradient, coef] = gradient_of_HSIC(A, B, kernel, sigma)
% GRADIENT_OF_HSIC used to compute the gradient of HSIC(A, B) w.r.t A,
% according to different kernels.
%
% Input:
%   A: matrix, n x d
%   B: matrix, n x l
%   kernel: string, "linear" or "gaussian"
%   sigma: double, used in constructing the gaussian kernel
%
% Output:
%   gradient: matrix, n x n, the gradient of HSIC(A, B) w.r.t A.
%
% Call:
%   [gradient] = gradient_of_HSIC(A, B, kernerl)(A, B, kernel)
%
% Version: 1.0, created on 03/16/2022, modified on 03/22/2022,
% Author: Zhiwei Li

if nargin < 3
    kernel = 'linear';
    sigma = sqrt(var(var(A)));
elseif nargin < 4
    sigma = sqrt(var(var(A)));
end

if ~strcmp(kernel, 'linear') && ~strcmp(kernel, 'gaussian')
    info = 'Wrong kernel! For now, RIMAR only supports linear or gaussian kernel!';
    disp(info);
end

% compute the gradient of HSIC(A, B) w.r.t. A according to different kernels
if strcmp(kernel, 'linear')
    HB = bsxfun(@minus, B, mean(B));
    coef = 2 * (HB * HB');
elseif strcmp(kernel, 'gaussian')

    n = size(A, 1); % the number of samples

    H = eye(n, n) - 1/n;

    Ka = construct_kernel(A, kernel, sigma);
    Kb = construct_kernel(B, kernel, sigma);

    % Q = H * Kb * H;
    Q = bsxfun(@minus, Kb, mean(Kb)) * H;

    % R = -1 / (2 * sigma^2) * Ka .* Q;
    R = bsxfun(@times, -1 / (2 * sigma^2) * Ka, Q);

    M = diag(R * eye(n, n)) - R;

    coef = -4 * M;
end

gradient = coef * A;

end
function[res] = HSIC(A, B, kernel, sigma)
% HSIC used to compute HSIC(A, B),
% according to different kernels.
%
% Input:
%   A: matrix, n x d
%   B: matrix, n x l
%   kernel: string, "linear" or "gaussian"
%   sigma: double, used in constructing the gaussian kernel
%
% Output:
%   res: scalar, the value of HSIC(A, B)
%
% Call:
%   [res] = HSIC(A, B, kernel)
%
% Version: 1.0, created on 03/15/2022, modified on 03/21/2022,
% Author: Zhiwei Li

% Set default values
if nargin < 3
    kernel = 'linear';
    sigma = 0;
elseif nargin < 4
    sigma = 0;
end

Ka = construct_kernel(A, kernel, sigma);
Kb = construct_kernel(B, kernel, sigma);

HKa = bsxfun(@minus, Ka, mean(Ka));
HKb = bsxfun(@minus, Kb, mean(Kb));

res = max(trace(HKa * HKb), eps);

end
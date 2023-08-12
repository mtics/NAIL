function[alpha] = line_search(f, d, x, alpha, direction)
% -------------------------------------------------------
%           Using Line Search to find                   |
%           the learing rate automatically.             |
% -------------------------------------------------------
%
% Input:
%   f:          the objective function
%   d:          the gradient function of objective w.r.t x
%   x:          the current point
%   alpha:      the current or initial learning rate
%   direction:  the gradient's direction
%
% Output:
%   alpha: the found learning rate
%
% Call:
%   [alpha] = line_search(fun, grad, x)
%   [alpha] = line_search(fun, grad, x, learing_rate)
%   [alphas] = line_search(fun, grad, x, learing_rate, direction)
%
% Note:
%   This method would be time-consuming
%
% Reference:
%   https://lumingdong.cn/setting-strategy-of-gradient-descent-learning-rate.html
%   《梯度下降学习率的设定策略》 
%                   -- 卢明冬的博客

% Set default values
if nargin == 3
    alpha = 1e-5;
    direction = -1;
elseif nargin == 4
    direction = -1;
end

gradient = d(x); % the gradient of the objective when xk = x

y = x + alpha * direction * gradient;           % The next point
T = @(yk, a)(yk + a * direction * d(yk));

%% Step 1. Find the largest learning rate first.
iter = 0;
while f(T(y, alpha)) <= f(y) + trace(d(y)' * (T(y, alpha) - y)) + norm(T(y, alpha) - y)^2 / (2 * alpha)
    alpha = alpha * 2;
    y = x + alpha * direction * gradient;
    iter = iter+1;
end

%% Step 2. Find the largest learning rate that satisifies the condition
count = 0;
while f(T(y, alpha)) > f(y) + trace(d(y)' * (T(y, alpha) - y)) + norm(T(y, alpha) - y)^2 / (2 * alpha)
    alpha = alpha / 10;
    y = x + alpha * direction * gradient;
    count = count+1;
end

% alpha = min(alpha, 1e0);

% fprintf("Line Search stops at %d-th iter. \n", count);

end
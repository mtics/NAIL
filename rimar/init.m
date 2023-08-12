function[O, F, U, alphas, betas, indexes, options] = init(Ox, Oy, X, Y, params)

rng('default');

%% Definition and Initialization
% Define local parameters
m = length(X);                  % number of feature views

alphas = ones(1, m) / m;
betas = ones(m+1, m+1) / m;         % balanced terms of HSIC

d = zeros(1, m+1);              % feature dimensions
[n, d(m+1)]= size(Y);           % number of samples and label dimension
O = zeros(m, n);                % indexes for observed features
for v = 1:m
    d(v) = size(X{v}, 2);
    O(v, (Ox{v}(:, 1)==1)) = 1;
end
k = min(floor(params.subRatio * d)); % subspace dimensions

F = rand(n, k);
U = cell(1, m+1);
for v = 1:m+1
    U{v} = rand(k, d(v));
end

% Construct the indexes for the observed labels
[rows, cols, labels] = find(Oy .* Y);
tempIndex = [rows, cols, labels];
negatives = tempIndex(tempIndex(:, 3) == -1, :);
positives = tempIndex(tempIndex(:, 3) == 1, :);

indexes = cell(2, d(m+1));
for col =  1:d(m+1)
    indexes{1, col} = negatives(negatives(:, 2) == col, 1); % store negative samples of the i-th label
    indexes{2, col} = positives(positives(:, 2) == col, 1); % store positive samples of the i-th label
end

% Construct the params used in Projected Gradient Descent
options = struct();
options.lr = params.lr;
options.epoch = params.epoch;
options.tol = params.tol;
options.isFixed = false;
options.normalize = false;

%% Optimization
convergencedIter = -1;
losses = zeros(4, params.maxIter);
for iter = 1:params.maxIter

    for v = 1:m+1
        %%% Update U
        objU = @(u)calc_obj_wrt_U(u, v, O, Oy, X, Y, F);
        gradU = @(u)calc_grad_wrt_U(u, v, O, Oy, X, Y, F);

        U{v} = projected_gradient_descent(objU, gradU, U{v}, options);
    end

    %%% Update F
    objF = @(f)calc_obj_wrt_F(f, O, Oy, X, Y, U, alphas, params);
    gradF = @(f)calc_grad_wrt_F(f, O, Oy, X, Y, U, alphas, params);

    F = projected_gradient_descent(objF, gradF, F, options);

    %%% Check Convergence
    maskedLosses = zeros(1, m);
    for v = 1:m
        maskedLosses(v) = norm(O(v, :) * (X{v} - F*U{v}), 'fro')^2;
    end

    losses(1, iter) = sum(alphas.^params.s .* maskedLosses);
    
    losses(2, iter) = params.lambda * norm(Oy .* (Y-F*U{m+1}), 'fro')^2;

    losses(4, iter) = sum(losses(1:3, iter));

    pstr = "Iteration: [%d/%d], First: %5.4f,\t Second: %5.4f,\t Third: %5.4f,\t Total: %5.4f\n";
    fprintf(pstr, iter, params.maxIter, losses(1, iter), losses(2, iter), losses(3, iter),losses(4, iter));

    if iter > 1 && abs(losses(4,iter) - losses(4,iter-1)) / abs(losses(4,iter-1)) < params.tol
        if convergencedIter == -1
            convergencedIter = iter;
        end
        break;
    end

    %%% Update alphas
    tempLosses = (params.s * maskedLosses).^(1/(1-params.s));
    alphas = tempLosses ./ sum(tempLosses);

    options.lr = options.lr * exp(- 0.5 / iter);

end

hatY = F * U{m+1};
evaluation = evaluate(hatY(params.trainNum+1:size(Y, 1), :), Y(params.trainNum+1:size(Y, 1), :));
evaluation;

end

function[grad] = calc_grad_wrt_U(u, v, O, Oy, X, Y, F)

m = length(X);

if v ~= m+1
    grad = -2*F' * diag(O(v, :)) * (X{v} - F * u);
else
    grad = -2*F' * (Oy .* (Y - F * u));
end

end


function[obj] = calc_obj_wrt_U(u, v, O, Oy, X, Y, F)

m = length(X);

if v ~= m+1
    obj = norm(O(v, :) * (X{v} - F * u), 'fro')^2;
else
    obj = norm(Oy .* (Y - F * u), 'fro')^2;
end

end


function[grad] = calc_grad_wrt_F(f, O, Oy, X, Y, U, alphas, params)

m = length(X);

grad = -2 * params.lambda * (Oy .* (Y - f*U{m+1})) * U{m+1}';

for v = 1:m
    grad = grad - 2 * alphas(v)^params.s * (diag(O(v, :)) * (X{v} - f*U{v})) * U{v}';
end

end


function[obj] = calc_obj_wrt_F(f, O, Oy, X, Y, U, alphas, params)

m = length(X);

obj = params.lambda * norm(Oy .* (Y - f*U{m+1}), 'fro')^2;

for v = 1:m
    obj = obj - alphas(v)^params.s * norm(diag(O(v, :)) * (X{v} - f*U{v}), 'fro')^2;
end

end

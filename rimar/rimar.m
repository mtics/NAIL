function [outputs] = rimar(Ox, Oy, X, Y, params)

addpath(fullfile('rimar', 'model_tools'));

% profile on

params.s = 0.5;
params.r = 2;
params.lr = 1e-6;
params.epoch = floor(params.maxIter/10);

% params.mu = 0;

fprintf("First, RIMAR runs a warm start initialize programm.\n");
[O, F, U, alphas, betas, indexes, options] = init(Ox, Oy, X, Y, params);
fprintf("The warm start initialize programm now is ended.\n\n");

% options.lr = params.lr;

m = length(X);

fprintf("The main optimization programm now is started.\n");

%% Optimization
convergencedIter = -1;
losses = zeros(4, params.maxIter);
for iter = 1:params.maxIter

    %% Update U
    for v = 1:m+1
        U{v} = update_U(U{v}, v, O, X, F, U, alphas, betas, indexes, params, options);
    end

    %% Update F
    F = update_F(F, O, X, U, alphas, indexes, params, options);

    %% Update related losses
    maskedLosses = zeros(1, m);
    HSICLosses = zeros(m+1, m+1);
    for v = 1:m+1
        if v ~= m+1
            dist = diag(pdist2(diag(O(v, :))*X{v}, diag(O(v, :))*F*U{v}));
            maskedLosses(v) = sum(dist);
        end
        
        for v2 = 1:m+1
            if v2 ~= v
                HSICLosses(v, v2) = HSIC(U{v}, U{v2}, params.kernel, params.sigma);
            end
        end
    end

    %% Check Convergence
    losses(1, iter) = sum(alphas.^params.s .* maskedLosses);
    losses(2, iter) = params.lambda * focal_loss(F, U{m+1}, indexes, params);
    losses(3, iter) = params.mu * sum(betas .* HSICLosses, 'all');
    losses(4, iter) = sum(losses(1:3, iter));

    pstr = "Iteration: [%d/%d], First: %5.4f,\t Second: %5.4f,\t Third: %5.4f,\t Total: %5.4f\n";
    fprintf(pstr, iter, params.maxIter, losses(1, iter), losses(2, iter), losses(3, iter),losses(4, iter));

    if iter > 1 && abs(losses(4,iter) - losses(4,iter-1)) / abs(losses(4,iter-1)) < params.tol
        if convergencedIter == -1
            convergencedIter = iter;
        end
        break;
    end

    %% Update alphas
    tempLosses = (params.s * maskedLosses).^(1/(1-params.s));
    alphas = tempLosses ./ sum(tempLosses);

    %% Update betas
    betas = sqrt(sum(HSICLosses.^0.5)) ./ max(HSICLosses.^0.5, eps);

    %% Update learning rate
    options.lr = options.lr * exp(- 0.5 / iter);

end

%% Ouput Results
outputs = struct();

outputs.losses = losses;
outputs.convergencedIter = convergencedIter;

outputs.alphas = alphas;
outputs.betas = betas;

outputs.hatY = 1 ./ (1+exp(-F*U{m+1}));
outputs.hatX = cell(1,m);
for v = 1:m
    outputs.hatX{v} = F*U{v};
end

outputs.F = F;
outputs.U = U;

% profile viewer
% p = profile('info');
% profsave(p, 'outputs/rimar/logs');
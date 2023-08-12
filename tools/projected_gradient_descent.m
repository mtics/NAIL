function[A, learningRate, losses] = projected_gradient_descent(objective, gradient, A, options)

learningRate = options.lr;

%% Gradient Descent
losses = zeros(1, options.epoch);
for run = 1:options.epoch

    % Use line search to set the step size
    if ~options.isFixed
        %         stepSize = line_search(objective, gradient, A, stepSize, -1);
        learningRate = min(line_search(objective, gradient, A, learningRate, -1), 1);
    end

    A = max(A - learningRate * gradient(A), 0);

    if options.normalize
        % Normalize
        A(isnan(A)) = 0;

        norms = sum(abs(A),1);
        norms = max(norms,eps);
        %F{v} = F{v} ./ repmat(norms, n, 1);
        A = bsxfun(@rdivide, A, repmat(norms, size(A, 1), 1));
    end

    % Compute the objective loss
    losses(run) = objective(A);

    if run > 1 && abs(losses(run) - losses(run-1)) / abs(losses(run-1)) < options.tol
        break;
    end
end

end
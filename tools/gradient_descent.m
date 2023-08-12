function[A, stepSize, losses] = gradient_descent(objective, gradient, A, options)

stepSize = options.stepSize;

%% Gradient Descent
losses = zeros(1, options.epoch);
for run = 1:options.epoch

    % Use line search to set the step size
    if ~options.isFixed
%         stepSize = line_search(objective, gradient, A, stepSize, -1);
        stepSize = min(line_search(objective, gradient, A, stepSize, -1), 1);
    end

    A = A - stepSize * gradient(A);

    % Compute the objective loss
    losses(run) = objective(A);

    if run > 1 && abs(losses(run) - losses(run-1)) / abs(losses(run-1)) < options.tol
        break;
    end
end

end
function[newu] = update_U(u, v, O, X, F, U, alphas, betas, indexes, params, options)

obj = @(x)obj_wrt_U(x, v, O, X, F, U, alphas, betas, indexes, params);
grad = @(x)grad_wrt_U(x, v, O, X, F, U, alphas, betas, indexes, params);

newu = projected_gradient_descent(obj, grad, u, options);

end


function[grad] = grad_wrt_U(u, v, O, X, F, U, alphas, betas, indexes, params)

m = length(X);

if v ~= m+1
    dist = max(diag(pdist2(diag(O(v, :))*X{v}, diag(O(v, :))*F*u)), eps);

    D = 1 ./ dist';

    grad = alphas(v)^params.s * F' * diag(D .* O(v, :)) * (X{v} - F * u);
else
    [~, grad] = gradient_of_FL(F, u, indexes, params);
end

for v2 = 1:m+1
    if v2 ~= v
        grad = grad + params.mu * betas(v, v2) * gradient_of_HSIC(u, U{v2}, params.kernel, params.sigma);
    end
end

end


function[val] = obj_wrt_U(u, v, O, X, F, U, alphas, betas, indexes, params)

m = length(X);

if v ~= m+1
    dist = diag(pdist2(diag(O(v, :))*X{v}, diag(O(v, :))*F*u));
    val = sum(alphas(v)^params.s * dist);
else
    val = params.lambda * focal_loss(F, u, indexes, params);
end

for v2 = 1:m+1
    if v2 ~= v
        val = val + params.mu * betas(v, v2) * HSIC(u, U{v2}, params.kernel, params.sigma);
    end
end

end
function[newf] = update_F(f, O, X, U, alphas, indexes, params, options)

obj = @(x)obj_wrt_F(x, O, X, U, alphas, indexes, params);
grad = @(x)grad_wrt_F(x, O, X, U, alphas, indexes, params);

newf = projected_gradient_descent(obj, grad, f, options);

end


function[grad] = grad_wrt_F(f, O, X, U, alphas, indexes, params)

m = length(X);

grad = zeros(size(f));
for v = 1:m
    dist = max(diag(pdist2(diag(O(v, :))*X{v}, diag(O(v, :))*f*U{v})), eps);
    D = 1 ./ dist';
    grad = grad - alphas(v)^params.s * diag(D .* O(v, :)) * (X{v} - f * U{v}) * U{v}';
end

[grad, ~] = gradient_of_FL(f, U{m+1}, indexes, params);

end


function[val] = obj_wrt_F(f, O, X, U, alphas, indexes, params)

m = length(X);

val = params.lambda * focal_loss(f, U{m+1}, indexes, params);

for v = 1:m
    dist = diag(pdist2(diag(O(v, :))*X{v}, diag(O(v, :))*f*U{v}));
    val = val + alphas(v)^params.s * sum(dist);
end

end
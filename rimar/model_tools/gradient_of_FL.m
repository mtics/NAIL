function[gradF, gradU] = gradient_of_FL(F, U, indexes, params)

gradF = zeros(size(F));
gradU = zeros(size(U));
for j = 1:size(U, 2)
    for row = 1:2

        i = indexes{row, j};

        p = 1 ./ (1 + exp(- F(i, :) * U(:, j)));

        y = 2*row - 3;

        q = get_pt(p, y);

        coef = y * (1 - q).^params.r .* (params.r .* q .* log(q) + q - 1);

        gradF(i, :) = gradF(i, :) + params.lambda * sum(coef * U(:, j)', 2);

        gradU(:, j) = gradU(:, j) + params.lambda * sum(F(i, :)' * coef);

    end
end
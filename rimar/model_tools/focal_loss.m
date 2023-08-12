function[objective] = focal_loss(F, U, indexes, params)

l = size(U, 2);

objective = 0;
for col = 1:l
    for row = 1:2
        
        rows = indexes{row, col};

        p = 1 ./ (1 + exp(-F(rows, :) * U(:, col)));

        y = 2*row - 3;

        q = get_pt(p, y);

        objective = objective - sum((1 - q) .^params.r .* log(q));

    end
end

end
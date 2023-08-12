function[betas] = update_betas(HSICLoss)


betas = sqrt(sum(HSICLoss.^2)) ./ HSICLosses;

end
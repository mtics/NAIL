function[value] = pnorm(X, p)

value = sum(sum(abs(X).^p))^(1/p);

end
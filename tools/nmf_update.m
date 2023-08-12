function[A] = nmf_update(A, C, D)

[Cp, Cn] = splite_numbers(C);
[Dp, Dn] = splite_numbers(D);

A = A .* sqrt((Cn + Dp) ./ (Cp + Dn + eps));

end

function[pos, neg] = splite_numbers(A)

pos = (abs(A) + A) / 2;
neg = (abs(A) - A) / 2;

end
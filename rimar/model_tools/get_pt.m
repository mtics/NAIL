function[pt] = get_pt(p, y)

pt = (y+1)/2 * p + (1-y)/2 * (1-p);
pt = min(max(pt, eps), 1-eps);

end
function[head_indexes, tail_indexes] = tail_label(labels, ratio)
% TAIL_LABEL plot the distribution of labels and return the descend order
% sorted indexes.
%
% Input:
%   labels: a 0/1 matrix.
%
% Output:
%   indexes: index the values sorted by descend order.
%   tail_ratio: set the last part of labels as tail labels
%
% Call:
%   [indexes] = tail_label(labels)

[n, l] = size(labels);
label_num = sum(labels, 1);
[sorted_num, sorted_indexes] = sort(label_num, 'descend');
head_indexes = sort(sorted_indexes(sorted_num > n*ratio));
tail_indexes = sort(sorted_indexes(sorted_num <= n*ratio));

figure;
plot(sorted_num, 'LineWidth', 2);
ylabel("Number of samples");
xlabel("Labels");
set(gca, 'FontSize', 18);

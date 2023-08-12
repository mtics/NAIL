function A=average_auroc(result,target)
% AVERAGE_AUROC used to compute the average auroc
%
% Input:
%   result: mat, the predicted result of the classifier.
%           The output of the i-th instance for the j-th class is stored in
%           result(j,i).If the i-th instance belong to the j-th class,
%           target(j,i) = 1, otherwise target(j,i) = -1
%   target: mat, the actual labels of the test instances.
%           If the i-th instance belong to the j-th class, target(j,i) = 1,
%           otherwise target(j,i) = -1
%
% Output:
%   A: the computed average auroc
%
% NOTICE: This function is collected from [1].
%         We only renamed the names of this function and its parameters,
%         and added some commentions.
%         ** We didn't change any implementation. **
%                                   -- Zhiwei Li
%
% Reference:
%   [1] Dong, Hao-Chen, Yu-Feng Li, and Zhi-Hua Zhou. 
%       "Learning from semi-supervised weak-label data." 
%       Proceedings of the AAAI Conference on Artificial Intelligence. 
%       Vol. 32. No. 1. 2018.

label_num=size(target,2);

A=zeros(label_num,1);
for aa=1:label_num
    if aa == 527
       ttt = 0; 
    end
    
    [tp,fp]=roc(target(:,aa),result(:,aa));
    
    A(aa,1)=auroc(tp,fp);
end

A=mean(A);
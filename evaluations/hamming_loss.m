function hammingLoss=hamming_loss(result,target)
% HAMMING_LOSS used to compute the hamming loss
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
%   hammingLoss: double, the computed hamming loss
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

[num_class,num_instance]=size(result);
%     Pre_Labels = ((Pre_Labels > 0)-.5)*2;
miss_pairs=sum(sum(result~=target));
hammingLoss=miss_pairs/(num_class*num_instance);

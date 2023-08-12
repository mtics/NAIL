function rankingLoss=ranking_loss(result,target)
% RANKING_LOSS used to compute the ranking loss
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
%   rankingLoss: double, the computed ranking loss
%
% NOTICE: This function is collected from [1]
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
temp_Outputs=[];
temp_test_target=[];
for i=1:num_instance
    temp=target(:,i);
    if((sum(temp)~=num_class)&(sum(temp)~=-num_class))
        temp_Outputs=[temp_Outputs,result(:,i)];
        temp_test_target=[temp_test_target,temp];
    end
end
result=temp_Outputs;
target=temp_test_target;
[num_class,num_instance]=size(result);

Label=cell(num_instance,1);
not_Label=cell(num_instance,1);
Label_size=zeros(1,num_instance);
for i=1:num_instance
    temp=target(:,i);
    Label_size(1,i)=sum(temp==ones(num_class,1));
    for j=1:num_class
        if(temp(j)==1)
            Label{i,1}=[Label{i,1},j];
        else
            not_Label{i,1}=[not_Label{i,1},j];
        end
    end
end

rankloss=0;
for i=1:num_instance
    temp=0;
    for m=1:Label_size(i)
        for n=1:(num_class-Label_size(i))
            if(result(Label{i,1}(m),i)<=result(not_Label{i,1}(n),i))
                temp=temp+1;
            end
        end
    end
    rl_binary(i)=temp/(m*n);
    rankloss=rankloss+temp/(m*n);
end
rankingLoss=rankloss/num_instance;
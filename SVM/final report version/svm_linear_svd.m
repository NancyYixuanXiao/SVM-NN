function [CCR] = svm_linear_svd(norm_method,feature,label,numoffold,startpoint_C,iter)
CCRi=zeros(1,numoffold);
c1 = cvpartition(label,'KFold',numoffold); % k folding
boxconstrain = startpoint_C*2^(iter-1);
for j = 1:numoffold
    trIdx = c1.training(j);
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
    svmpara = svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
        boxconstrain*ones(length(trainlabel),1),'tolkkt',2^-16,'options',...
        statset('Display','off','MaxIter',10^7),'kernelcachelimit',1000000,...
        'method',norm_method);
    result=svmclassify(svmpara,xtest);
    
    % performance calculation
    CCRi(1,j)=length(find((testlabel-result)==0))/length(testlabel);
end
CCR= mean(CCRi);

end

function [CCR] = svm_func_final(kernel,norm_method,feature,label,sigma,C) 
CCRi=zeros(1,10);
c1 = cvpartition(label,'KFold',10); % k folding
for j = 1:10
    trIdx = c1.training(j);
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
    svmpara=svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
        C,'kernel_function',kernel,'tolkkt',2^-7,'options',...
        statset('Display','off','MaxIter',10^7),'rbf_sigma',sigma,...
        'kernelcachelimit',10000000,'method',norm_method);
    result=svmclassify(svmpara,xtest);
    % performance calculation
    CCRi(1,j)=length(find((testlabel-result)==0))/length(testlabel);
end
CCR = mean(CCRi);
end
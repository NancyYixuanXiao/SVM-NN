function [CCR] = svm_func_1_final(kernel,norm_method,feature,label,sigma,C) 
warning('off','all')
CCRi=zeros(1,10);
c1 = cvpartition(label,'KFold',10); % k folding
for j = 1:10
    trIdx = c1.training(j);
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
    if strcmp(norm_method,'SMO') == 1
        svmpara=fitcsvm(xtrain,trainlabel,'CacheSize',...
            1000000,'NumPrint',10^7,'KernelFunction',kernel,'BoxConstraint',C);
        result=predict(svmpara,xtest);
    else svmpara=svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
            C,'kernel_function',kernel,'tolkkt',2^-4.5,'options',...
            statset('Display','off','MaxIter',10^7),'rbf_sigma',sigma,...
            'kernelcachelimit',10000000,'method',norm_method);
        result=svmclassify(svmpara,xtest);
    end
    
    % performance calculation
    CCRi(1,j)=length(find((testlabel-result)==0))/length(testlabel);
end
CCR = mean(CCRi);
end
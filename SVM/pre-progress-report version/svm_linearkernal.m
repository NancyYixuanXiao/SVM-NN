function [CCR,prec,recall,fscore] = svm_linearkernal(norm_method,feature,label,numoffold,boxconstrain)
CCRi=zeros(1,numoffold); preci=zeros(1,numoffold); 
recalli=zeros(1,numoffold); fscorei=zeros(1,numoffold); 
c1 = cvpartition(label,'KFold',numoffold); % k folding
for j = 1:numoffold
    trIdx = c1.training(j);
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
    svmpara = svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
        boxconstrain*ones(length(trainlabel),1),'tolkkt',2^-16,'options',...
        statset('Display','off','MaxIter',10^7),'kernelcachelimit',1000000,...
        'method',norm_method);
%     svmpara=fitcsvm(xtrain,trainlabel,'method',norm_method,'CacheSize',...
%         500000,'NumPrint',150000,'KernelFunction','linear','BoxConstraint',boxconstrain);
    result=svmclassify(svmpara,xtest);

    % performance calculation
    conf=confusionmat(testlabel,result);
    preci(1,j)=conf(1,1)/(conf(1,1)+conf(2,1));
    recalli(1,j)=conf(1,1)/(conf(1,1)+conf(1,2));
    fscorei(1,j)=2*preci(1,j)*recalli(1,j)/(preci(1,j)+recalli(1,j));
    CCRi(1,j)=length(find((testlabel-result)==0))/length(testlabel);
end
CCR = mean(CCRi); prec=mean(preci);
recall=mean(recalli);
fscore=mean(fscorei);
end

%     svmpara1 = svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
%         boxconstrain*ones(length(trainlabel),1),'tolkkt',0.05,'options',...
%         statset('Display','off','MaxIter',10^7),'kernelcachelimit',500000);
%     result=svmclassify(svmpara1,xtest);
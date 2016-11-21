function [CCR,prec,recall,fscore] = ...
    svm_polykernal(norm_method,feature,label,numoffold,boxconstrain,numofpoly)
CCR=zeros(1,numofpoly); prec=zeros(1,numofpoly); 
recall=zeros(1,numofpoly); fscore=zeros(1,numofpoly); 
CCRi=zeros(1,numoffold); preci=zeros(1,numoffold); 
recalli=zeros(1,numoffold); fscorei=zeros(1,numoffold); 
c1 = cvpartition(label,'KFold',numoffold); % k folding
for k = 1:numofpoly
    for j = 1:numoffold
        trIdx = c1.training(j);
        xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
        xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
        svmpara = svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
        boxconstrain*ones(length(trainlabel),1),'tolkkt',2^-10,'options',...
        statset('Display','off','MaxIter',10^7),'kernelcachelimit',1000000,...
        'method',norm_method,'kernel_function','polynomial','polyorder',k);
        result=svmclassify(svmpara,xtest);
        % performance calculation
        conf=confusionmat(testlabel,result);
        preci(1,j)=conf(1,1)/(conf(1,1)+conf(2,1));
        recalli(1,j)=conf(1,1)/(conf(1,1)+conf(1,2));
        fscorei(1,j)=2*preci(1,j)*recalli(1,j)/(preci(1,j)+recalli(1,j));
        CCRi(1,j)=length(find((testlabel-result)==0))/length(testlabel);
    end
    CCR(1,k) = mean(CCRi); 
    prec(1,k)=mean(preci);
    recall(1,k)=mean(recalli);
    fscore(1,k)=mean(fscorei);
end


end
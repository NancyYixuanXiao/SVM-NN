function [CCR,prec,recall,fscore] = svm_linear_cost(feature,label,numoffold,boxconstrain,fpcost,fncost)
CCRi=zeros(1,numoffold); preci=zeros(1,numoffold); 
recalli=zeros(1,numoffold); fscorei=zeros(1,numoffold); 
c1 = cvpartition(label,'KFold',numoffold); % k folding
for j = 1:numoffold
    trIdx = c1.training(j);
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
    svmpara=fitcsvm(xtrain,trainlabel,'Cost',[0,fncost;fpcost,0],'CacheSize',...
        500000,'NumPrint',150000,'BoxConstraint',boxconstrain);
    result=predict(svmpara,xtest);

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
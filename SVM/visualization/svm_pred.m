function [CCR,svmStruct,xtrain,trainlabel,xtest,result] = svm_pred(kernel,norm_method,sigma,C,feature,label)


% warning('off','all')
c1 = cvpartition(label,'KFold',3); 

if strcmp(kernel,'rbf') == 1
    trIdx = c1.training(1);
    feature = bsxfun(@minus,feature,mean(feature));
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
    coef = pca(xtrain);
    xtrain = xtrain*coef(:,1:3); xtest = xtest*coef(:,1:3);
    svmStruct=svmtrain(xtrain,trainlabel,'showplot','false','kernel_function',kernel,...
      'boxconstraint',C,'kktviolationlevel',0.05,'tolkkt',5e-3,'method',norm_method...
      ,'rbf_sigma',sigma);
    result=svmclassify(svmStruct,xtest);
    plot3(xtrain(trainlabel==2,1),xtrain(trainlabel==2,2),xtrain(trainlabel==2,3),'r.','MarkerSize',12); hold on
    plot3(xtrain(trainlabel==1,1),xtrain(trainlabel==1,2),xtrain(trainlabel==1,3),'b.','MarkerSize',12);
    
    CCR=length(find((testlabel-result)==0))/length(testlabel);
end

if strcmp(kernel,'linear') == 1
    trIdx = c1.training(1);
    feature = bsxfun(@minus,feature,mean(feature));
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
    coef = pca(xtrain);
    xtrain = xtrain*coef(:,1:3); xtest = xtest*coef(:,1:3);
    svmStruct=svmtrain(xtrain,trainlabel,'showplot','false','kernel_function',kernel,...
      'boxconstraint',C,'kktviolationlevel',0.05,'tolkkt',5e-3,'method',norm_method...
      ,'rbf_sigma',sigma);
    result=svmclassify(svmStruct,xtest);
    
    plot3(xtrain(trainlabel==2,1),xtrain(trainlabel==2,2),xtrain(trainlabel==2,3),'r.','MarkerSize',12); hold on
    plot3(xtrain(trainlabel==1,1),xtrain(trainlabel==1,2),xtrain(trainlabel==1,3),'b.','MarkerSize',12);
    
    CCR=length(find((testlabel-result)==0))/length(testlabel);
end







end
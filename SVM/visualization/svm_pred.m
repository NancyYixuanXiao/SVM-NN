function [CCR,svmStruct,xtrain,trainlabel,xtest,testlabel] = svm_pred(kernel,norm_method,sigma,C,feature,label)


warning('off','all')
c1 = cvpartition(label,'KFold',3); 

if strcmp(kernel,'rbf') == 1
    trIdx = c1.training(1);
    feature = bsxfun(@minus,feature,mean(feature));
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
    [V,~]=eig(xtrain'*xtrain);
    xtraint=xtrain*V; xtrain=xtraint(:,end-2:end);
    xtestt=xtest*V; xtest=xtestt(:,end-2:end);
    svmStruct=svmtrain(xtrain,trainlabel,'showplot','false','kernel_function',kernel,...
      'boxconstraint',C,'tolkkt',2^(-14),'method',norm_method,'options',...
                    statset('Display','off','MaxIter',10^7)...
      ,'rbf_sigma',sigma);
    result=svmclassify(svmStruct,xtest);
    fig1=figure;
    plot3(xtrain(trainlabel==1,1),xtrain(trainlabel==1,2),xtrain(trainlabel==1,3),'r.','MarkerSize',12); hold on
    plot3(xtrain(trainlabel==0,1),xtrain(trainlabel==0,2),xtrain(trainlabel==0,3),'b.','MarkerSize',12);
    svm_3d_vis(svmStruct,xtrain,trainlabel,1)
    axis([-1 2 -1 1 -0.5 1]); legend('positive','negative');
    fig2=figure;
    plot3(xtest(testlabel==1,1),xtest(testlabel==1,2),xtest(testlabel==1,3),'r.','MarkerSize',12); hold on
    plot3(xtest(testlabel==0,1),xtest(testlabel==0,2),xtest(testlabel==0,3),'b.','MarkerSize',12);
    svm_3d_vis(svmStruct,xtest,testlabel,0)
    axis([-1 2 -1 1 -0.5 1]); legend('positive','negative');
    
    CCR=length(find((testlabel-result)==0))/length(testlabel);
end

if strcmp(kernel,'linear') == 1
    trIdx = c1.training(2);
    feature = bsxfun(@minus,feature,mean(feature));
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
    [V,~]=eig(xtrain'*xtrain);
    xtraint=xtrain*V; xtrain=xtraint(:,end-2:end);
    xtestt=xtest*V; xtest=xtestt(:,end-2:end); % change num of feature you wanna use
    
    svmStruct=svmtrain(xtrain,trainlabel,'showplot','false','kernel_function',kernel,...
      'boxconstraint',C,'tolkkt',2^(-3),'method',norm_method);
    result=svmclassify(svmStruct,xtest);
    
    fig1=figure;
    plot3(xtrain(trainlabel==1,1),xtrain(trainlabel==1,2),xtrain(trainlabel==1,3),'r.','MarkerSize',12); hold on
    plot3(xtrain(trainlabel==0,1),xtrain(trainlabel==0,2),xtrain(trainlabel==0,3),'b.','MarkerSize',12);
    svm_3d_vis(svmStruct,xtrain,trainlabel,1)
    axis([-1 2 -1 1 -0.5 1]); legend('positive','negative');
    fig2=figure;
    plot3(xtest(testlabel==1,1),xtest(testlabel==1,2),xtest(testlabel==1,3),'r.','MarkerSize',12); hold on
    plot3(xtest(testlabel==0,1),xtest(testlabel==0,2),xtest(testlabel==0,3),'b.','MarkerSize',12);
    svm_3d_vis(svmStruct,xtest,testlabel,0)
    axis([-1 2 -1 1 -0.5 1]); legend('positive','negative');
    
    CCR=length(find((testlabel-result)==0))/length(testlabel);
end







end
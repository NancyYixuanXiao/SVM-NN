clear
% [feature,label] = svm_cell2mat(noli,setop,corefeat);
[feature,label] = svm_cell2mat(1,2,0);

% plot correlation coefficient
% corcoef = corrcoef(feature);
% mesh(1:30,1:30,corcoef);
% [x,y]=find(corcoef>0.7 & corcoef<1);
% scatter(x,y);

[CCR,svmStruct,xtrain,trainlabel,xtest,result] = svm_pred('rbf','SMO',0.125,1,feature,label);

svm_3d_vis(svmStruct,xtest,result)


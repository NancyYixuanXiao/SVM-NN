clear
% [feature,label] = svm_cell2mat(noli,setop,corefeat);
[feature,label] = svm_cell2mat(1,2,0);

% plot correlation coefficient
corcoef = corrcoef(feature);
contourf(1:32,1:32,corcoef); hold on
[x,y]=find(corcoef>0.95 & corcoef<1);
scatter(x,y,'r','LineWidth',1.5); hold off

% % plot 3d data points with hyper plane
% [CCR,svmStruct,xtrain,trainlabel,xtest,result] = svm_pred('linear','QP',0.125,1,feature,label);
% legend('positive','negative');

% % plot distribution of data
% figure(1);
% boxplot(feature(label==1,:),'Colors','r','Whisker',3);hold on 
% boxplot(feature(label==2,:),'Colors','b','Whisker',3); hold off
% title('boxplot of WPBC with normalization');


clear
% preprocessing csv to mat
[feature,label,~,~] = svm_cell2mat();
[U,S,V] = svds(feature,3);

[CCR] = svm_3d_matlab_vis(U,label);

% plot correlation coefficient
% corcoef = corrcoef(feature);
% mesh(1:30,1:30,corcoef);
% [x,y]=find(corcoef>0.7 & corcoef<1);
% scatter(x,y);


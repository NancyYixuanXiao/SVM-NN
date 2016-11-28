clear
% preprocessing csv to mat
[feature,label] = svm_csv2mat();

new_feat = (feature-repmat(mean(feature),[569 1]))./repmat(std(feature),[569 1]);
% plot(var(new_feat));
crosscov = corrcoef(feature);
contourf(1:30,1:30,crosscov); colorbar
title('corelation coefficient of each feature in data');
xlabel('feature'); ylabel('feature'); 










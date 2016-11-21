clear
% preprocessing csv to mat
[feature,label] = svm_csv2mat();

[U,S,V] = svds(feature,15);
feat_new =U*S*V';

[CCR1,prec1,recall1,fscore1] = svm_linearkernal(feature,label,10,1);
[CCR2,prec2,recall2,fscore2] = svm_rbfkernal(feature,label,10,2^10,2^8,1);
[CCR3,prec3,recall3,fscore3] = svm_linearkernal(feat_new,label,10,0.5);
[CCR4,prec4,recall4,fscore4] = svm_rbfkernal(feat_new,label,10,2^12,2^9,1);






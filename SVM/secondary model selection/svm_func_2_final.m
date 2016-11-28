function [CCR_total] = svm_func_2_final()
% preprocessing csv to mat
[feature,label] = svm_csv2mat();

% number of iterations in CV
numofiter = 8;

% prepare data for SVD
[U,S,V] = svds(feature,27); feat_svd_1 =U*S*V';
[U,S,V] = svds(feature,24); feat_svd_2 =U*S*V';
[U,S,V] = svds(feature,21); feat_svd_3 =U*S*V';

CCR_total_1 = zeros(1,numofiter);
CCR_total_2 = zeros(1,numofiter);
CCR_total_3 = zeros(1,numofiter);
CCR_total_4 = zeros(1,numofiter);
CCR_total_5 = zeros(1,numofiter);
CCR_total_6 = zeros(1,numofiter);
CCR_total_7 = zeros(1,numofiter);
CCR_total_8 = zeros(1,numofiter);
parfor i = 1:8
    C_lin = 2^(2*i-6);
    C_rbf = 2^(i+8);
        [CCR1] = svm_func_1_final('linear','SMO',feature,label,0,C_lin);
        [CCR2] = svm_func_1_final('linear','QP',feature,label,0,C_lin);
        [CCR3] = svm_func_1_final('rbf','QP',feature,label,2^8,C_rbf);
        [CCR4] = svm_func_1_final('rbf','QP',feature,label,2^10,C_rbf);
        [CCR5] = svm_func_1_final('rbf','QP',feature,label,2^12,C_rbf);
        [CCR6] = svm_func_1_final('linear','QP',feat_svd_1,label,0,C_lin);
        [CCR7] = svm_func_1_final('linear','QP',feat_svd_2,label,0,C_lin);
        [CCR8] = svm_func_1_final('linear','QP',feat_svd_3,label,0,C_lin);
        CCR_total_1(1,i) = CCR1;
        CCR_total_2(1,i) = CCR2;
        CCR_total_3(1,i) = CCR3;
        CCR_total_4(1,i) = CCR4;
        CCR_total_5(1,i) = CCR5;
        CCR_total_6(1,i) = CCR6;
        CCR_total_7(1,i) = CCR7;
        CCR_total_8(1,i) = CCR8;
end
% row represent model, column represetn parameters
CCR_total = [CCR_total_1;CCR_total_2;CCR_total_3;CCR_total_4...
    ;CCR_total_5;CCR_total_6;CCR_total_7;CCR_total_8];
end
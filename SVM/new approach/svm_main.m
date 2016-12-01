function svm_main
% multimodel SVM for EC503 project @ Boston university
% date:12/1/2016
%
% explain of functions:
% svm_main: read, preprocess and test data, output CCR plot of models
% svm_cell2mat: read data from csv files
% svm_3d_matlab_vis: visualize data for user
% svm_models: test preprocessed data on 4 SVM models (polynomial and
%     non-uniform cost model are optional)
% svm_func: SVM train and testing for models

% read in two data sets
[feature_wdbc,label_wdbc,feature_wpdc,label_wpdc] = svm_cell2mat();

% data preprocessing
feature = feature_wdbc;
label = label_wdbc;

% testing on multiple SVM models
svm_models(feature,label);
end
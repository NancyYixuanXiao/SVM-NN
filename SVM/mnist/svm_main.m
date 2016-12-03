% function svm_main(noli,setop,corefeat,featrd)
% multimodel SVM for EC503 project @ Boston university
% date:12/1/2016
%
% explain of functions:
% svm_main: read, preprocess and test data, output CCR plot of models
% svm_cell2mat & svm_csv2cell: read data from csv files
% svm_models: test preprocessed data on 4 SVM models (polynomial and
%     non-uniform cost model are optional)
% svm_func: SVM train and testing for models

% read in two data sets
load('data_mnist_train.mat');
load('data_mnist_test.mat');
feature = [X_train(1:12665,:);X_test(1:2115,:)];
label = [zeros(12665,1);ones(2115,1)];

% testing on multiple SVM models
svm_model_construct(feature,label,0);
% end
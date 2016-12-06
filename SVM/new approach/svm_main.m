function svm_main(noli,setop,corefeat,featrd)
% multimodel SVM for EC503 project @ Boston university
% date:12/3/2016
%
% explain of functions:
% svm_main: read, preprocess and test data, output CCR plot of models; 
%     noli = 1: use normalized data
%     setop = 1: wdbc; 2: wpbc
%     corefeat = 1: use manualy selected features
%     featrd = 1: use feature reduction
% svm_cell2mat & svm_csv2cell: read data from csv files
% svm_model_construct: test preprocessed data on 4 SVM models (polynomial and
%     non-uniform cost model are optional)
% svm_train_test: SVM train and testing for models

% read in two data sets
[feature,label] = svm_cell2mat(noli,setop,corefeat);
assignin('base','feature',feature);
assignin('base','label',label);

% testing on multiple SVM models
svm_model_construct(feature,label,featrd);
end
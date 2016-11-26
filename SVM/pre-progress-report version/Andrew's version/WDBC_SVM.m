% Load in WDBC dataset
disp('Loading Data')
data = csvread('WDBC_processed.csv');

% Filter dataset into input features and output label
disp('Filter Input and Output Data')
train_data = data(:,1:30);
train_label = data(:,31);

% options = statset('Display','off','MaxIter',10^9);
% SVMStruct = svmtrain(train_data,train_label,'boxconstraint',2^-8,'autoscale','false','kernel_function','linear','method','SMO','options',options);
% predict_labels = svmclassify(SVMStruct,train_data);
% conf_mat = confusionmat(train_label,predict_labels);
% CCR = trace(conf_mat) / length(train_label);

% Perform cross-validation to determine optimal cost penalty
num_fold = 5;
disp('Perform Cross-Validation')
CV_CCR_vec = WDBC_perform_CV(train_data,train_label,num_fold);

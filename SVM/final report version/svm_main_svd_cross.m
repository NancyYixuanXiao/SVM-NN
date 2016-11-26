clear
% preprocessing csv to mat
[feature,label] = svm_csv2mat();

%% enter parameters

% which regulization method you want to use
% 1 is L1 morm, 2 is L2 norm
norm = 2;

% number of k fold
numoffold = 10; 

% parameters for ALL kernal SVM
numofvalid = 5; % number of cross validation for C value 
which_C_to_start = 4; % 2^(which_C_to_start), minimum is -9
startpoint_C = 2^(which_C_to_start); % which boxconstrain to start with

if norm ==1
    norm_method = 'SMO';
elseif norm == 2
        norm_method = 'QP';
else 
    disp('please choose between L1 and L2 norm');
    return
end

if which_C_to_start < -15
    disp('C must be greater than 2^-15')
    return
end

% SVD
feat_new1 =feature;
[U2,S2,V2] = svds(feature,25); feat_new2 =U2*S2*V2'; % SVD
[U3,S3,V3] = svds(feature,20); feat_new3 =U3*S3*V3'; % SVD
[U4,S4,V4] = svds(feature,15); feat_new4 =U4*S4*V4'; % SVD

%% Linear Kernel
CCRlin1 = [];
parfor i = 1:numofvalid
    % cross validate on SVD and L1/L2 norm with linear kernel
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4];
    CCRlin1 = [CCRlin1 CCR_i];
end

CCRlin2 = [];
parfor i = 1:numofvalid
    % cross validate on SVD and L1/L2 norm with linear kernel
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4];
    CCRlin2 = [CCRlin2 CCR_i];
end

CCRlin3 = [];
parfor i = 1:numofvalid
    % cross validate on SVD and L1/L2 norm with linear kernel
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4];
    CCRlin3 = [CCRlin3 CCR_i];
end

CCRlin4 = [];
parfor i = 1:numofvalid
    % cross validate on SVD and L1/L2 norm with linear kernel
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4];
    CCRlin4 = [CCRlin4 CCR_i];
end

CCRlin5 = [];
parfor i = 1:numofvalid
    % cross validate on SVD and L1/L2 norm with linear kernel
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4];
    CCRlin5 = [CCRlin5 CCR_i];
end

CCRlin6 = [];
parfor i = 1:numofvalid
    % cross validate on SVD and L1/L2 norm with linear kernel
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4];
    CCRlin6 = [CCRlin6 CCR_i];
end

CCRlin7 = [];
parfor i = 1:numofvalid
    % cross validate on SVD and L1/L2 norm with linear kernel
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4];
    CCRlin7 = [CCRlin7 CCR_i];
end

CCRlin8 = [];
parfor i = 1:numofvalid
    % cross validate on SVD and L1/L2 norm with linear kernel
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4];
    CCRlin8 = [CCRlin8 CCR_i];
end

CCRlin9 = [];
parfor i = 1:numofvalid
    % cross validate on SVD and L1/L2 norm with linear kernel
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4];
    CCRlin9 = [CCRlin9 CCR_i];
end

CCRlin10 = [];
parfor i = 1:numofvalid
    % cross validate on SVD and L1/L2 norm with linear kernel
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4];
    CCRlin10 = [CCRlin10 CCR_i];
end

% calculate the average over 3 runs
CCRlin = (CCRlin1+CCRlin2+CCRlin3+CCRlin4+CCRlin5+CCRlin6+CCRlin7+CCRlin8+CCRlin9+CCRlin10)/10;

% find best C for optimal CCR, precision, recall and f-score
[~,maxC_CCR]=find(CCRlin==max(CCRlin(:)));

% plotting
fig1 = figure;
sk = (log(startpoint_C)/log(2)); % shift ploting start point
plot(sk:sk-1+numofvalid,CCRlin(1,:),sk:sk-1+numofvalid,CCRlin(2,:),sk:sk-1+numofvalid,...
    CCRlin(3,:),sk:sk-1+numofvalid,CCRlin(4,:)); hold on
plot(maxC_CCR+which_C_to_start-1,CCRlin(1,maxC_CCR),'x','MarkerSize',40); % mark optimal CCR
xlabel('C = 2^c'); ylabel('CCR');
legend('no svd','25 features','20 features','15 features');






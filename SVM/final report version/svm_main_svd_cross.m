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
numofvalid = 1; % number of cross validation for C value 
which_C_to_start = -15; % 2^(which_C_to_start), minimum is -9
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

%% Linear Kernel
CCRlin1 = [];
parfor i = 1:6
    % cross validate on SVD and L1/L2 norm with linear kernel
    feat_new1 =feature;
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [U2,S2,V2] = svds(feature,27); feat_new2 =U2*S2*V2'; % SVD
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [U3,S3,V3] = svds(feature,24); feat_new3 =U3*S3*V3'; % SVD
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [U4,S4,V4] = svds(feature,21); feat_new4 =U4*S4*V4'; % SVD
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    [U5,S5,V5] = svds(feature,18); feat_new5 =U5*S5*V5'; % SVD
    [CCR5] = svm_linear_svd(norm_method,feat_new5,label,numoffold,startpoint_C,i);
    [U6,S6,V6] = svds(feature,15); feat_new6 =U6*S6*V6'; % SVD
    [CCR6] = svm_linear_svd(norm_method,feat_new6,label,numoffold,startpoint_C,i);
%     [U7,S7,V7] = svds(feature,18); feat_new7 =U7*S7*V7'; % SVD
%     [CCR7] = svm_linear_svd(norm_method,feat_new7,label,numoffold,startpoint_C,i);
%     [U8,S8,V8] = svds(feature,16); feat_new8 =U8*S8*V8'; % SVD
%     [CCR8] = svm_linear_svd(norm_method,feat_new8,label,numoffold,startpoint_C,i);
%     [U9,S9,V9] = svds(feature,14); feat_new9 =U9*S9*V9'; % SVD
%     [CCR9] = svm_linear_svd(norm_method,feat_new9,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4;CCR5;CCR6;CCR7;CCR8;CCR9];
    CCRlin1 = [CCRlin1 CCR_i];
end

CCRlin2 = [];
parfor i = 1:6
    % cross validate on SVD and L1/L2 norm with linear kernel
    feat_new1 =feature;
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [U2,S2,V2] = svds(feature,27); feat_new2 =U2*S2*V2'; % SVD
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [U3,S3,V3] = svds(feature,24); feat_new3 =U3*S3*V3'; % SVD
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [U4,S4,V4] = svds(feature,21); feat_new4 =U4*S4*V4'; % SVD
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    [U5,S5,V5] = svds(feature,18); feat_new5 =U5*S5*V5'; % SVD
    [CCR5] = svm_linear_svd(norm_method,feat_new5,label,numoffold,startpoint_C,i);
    [U6,S6,V6] = svds(feature,15); feat_new6 =U6*S6*V6'; % SVD
    [CCR6] = svm_linear_svd(norm_method,feat_new6,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4;CCR5;CCR6;CCR7;CCR8;CCR9];
    CCRlin2 = [CCRlin2 CCR_i];
end

CCRlin3 = [];
parfor i = 1:6
    % cross validate on SVD and L1/L2 norm with linear kernel
    feat_new1 =feature;
    [CCR1] = svm_linear_svd(norm_method,feat_new1,label,numoffold,startpoint_C,i);
    [U2,S2,V2] = svds(feature,27); feat_new2 =U2*S2*V2'; % SVD
    [CCR2] = svm_linear_svd(norm_method,feat_new2,label,numoffold,startpoint_C,i);
    [U3,S3,V3] = svds(feature,24); feat_new3 =U3*S3*V3'; % SVD
    [CCR3] = svm_linear_svd(norm_method,feat_new3,label,numoffold,startpoint_C,i);
    [U4,S4,V4] = svds(feature,21); feat_new4 =U4*S4*V4'; % SVD
    [CCR4] = svm_linear_svd(norm_method,feat_new4,label,numoffold,startpoint_C,i);
    [U5,S5,V5] = svds(feature,18); feat_new5 =U5*S5*V5'; % SVD
    [CCR5] = svm_linear_svd(norm_method,feat_new5,label,numoffold,startpoint_C,i);
    [U6,S6,V6] = svds(feature,15); feat_new6 =U6*S6*V6'; % SVD
    [CCR6] = svm_linear_svd(norm_method,feat_new6,label,numoffold,startpoint_C,i);
    
    % record result on CCR, precision, recall and f-score
    CCR_i= [CCR1;CCR2;CCR3;CCR4;CCR5;CCR6;CCR7;CCR8;CCR9];
    CCRlin3 = [CCRlin3 CCR_i];
end

% calculate the average over 3 runs
CCRlin = (CCRlin1+CCRlin2+CCRlin3)/3;

% find best C for optimal CCR, precision, recall and f-score
[~,maxC_CCR]=find(CCRlin==max(CCRlin(:)));

% plotting
fig1 = figure;
sk = (log(startpoint_C)/log(2)); % shift ploting start point
plot(sk:sk-1+numofvalid,CCRlin); hold on
plot(maxC_CCR+which_C_to_start-1,CCRlin(1,maxC_CCR),'x','MarkerSize',40); % mark optimal CCR
xlabel('C = 2^c'); ylabel('CCR');






clear
% preprocessing csv to mat
[feature,label] = svm_csv2mat();

meanoffeat = mean(feature);
varoffeat = var(feature);

% SVD, but it is not used in this section, decision pending
[U,S,V] = svds(feature,15);
feat_new =U*S*V';

% number of k fold
numoffold = 10; 

% parameters for ALL kernal SVM
numofvalid = 24; % number of cross validation for C value 
if numofvalid/4-round(numofvalid/4) ~= 0
    disp('number of validation must be factor of 4');
    return
end
which_C_to_start = -9; % 2^(which_C_to_start)
startpoint_C = 2^(which_C_to_start); % which boxconstrain to start with

%% Linear Kernel cross validation
CCRlin = []; preclin = []; recalllin = []; fscorelin = [];
parfor i = 1:numofvalid
    iter = startpoint_C*2^(i-1);
    % cross validate on FN cost
    [CCR1,prec1,recall1,fscore1] = svm_linear_cost(feature,label,numoffold,iter,1,1.25);
    [CCR2,prec2,recall2,fscore2] = svm_linear_cost(feature,label,numoffold,iter,1,1.5);
    [CCR3,prec3,recall3,fscore3] = svm_linear_cost(feature,label,numoffold,iter,1,1.75);
    [CCR4,prec4,recall4,fscore4] = svm_linear_cost(feature,label,numoffold,iter,1,2);
    % record performance matrix
    CCRi= [CCR1; CCR2; CCR3; CCR4]; CCRlin = [CCRlin CCRi];
    preci= [prec1; prec2; prec3; prec4]; preclin = [preclin preci];
    recalli= [recall1; recall2; recall3; recall4]; recalllin = [recalllin recalli];
    fscorei= [fscore1; fscore2; fscore3; fscore4]; fscorelin = [fscorelin fscorei];
end
% find best C for optimal CCR, precision, recall and f-score
[~,maxC_CCR_lin]=find(CCRlin==max(CCRlin(:)));
[~,maxC_prec_lin]=find(preclin==max(preclin(:)));
[~,maxC_rec_lin]=find(recalllin==max(recalllin(:)));
[~,maxC_fsc_lin]=find(fscorelin==max(fscorelin(:)));
% plotting on each FN cost
fig1 = figure;
sk = (log(startpoint_C)/log(2));
subplot(2,2,1); plot(sk:sk-1+numofvalid,CCRlin); legend('1.25','1.5','1.75','2');
xlabel('C = 2^c'); ylabel('preformance matrics'); title('CCR');
subplot(2,2,2); plot(sk:sk-1+numofvalid,preclin); legend('1.25','1.5','1.75','2');
xlabel('C = 2^c'); ylabel('preformance matrics'); title('precision');
subplot(2,2,3); plot(sk:sk-1+numofvalid,recalllin); legend('1.25','1.5','1.75','2');
xlabel('C = 2^c'); ylabel('preformance matrics'); title('recall');
subplot(2,2,4); plot(sk:sk-1+numofvalid,fscorelin); legend('1.25','1.5','1.75','2');
xlabel('C = 2^c'); ylabel('preformance matrics'); title('f-score');
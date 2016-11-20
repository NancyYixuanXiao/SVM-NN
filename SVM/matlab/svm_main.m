clear
% preprocessing csv to mat
[feature,label] = svm_csv2mat();

%% enter parameters
% parameters for all svm model
numoffold = 10; % number of k fold

% parameters for linear kernal SVM
numofvalid = 24; % number of cross validation for  c value linear kernal
if numofvalid/4-round(numofvalid/4) ~= 0
    disp('number of validation must be factor of 4');
    return
end
startpoint_C = 2^(-9); % which boxconstrain to start with

% parameters for RBF kernal SVM
numofsigma = 24; % number of sigma to cross validate
startpoint_sigma = 2^(-1); % which sigma to start with

% parameters for polynomial kernal SVM
numofpoly = 24; % 2^(numofpoly-4), number of polynomial degree to cross validate


% %% linear kernal with k-fold
% CCRlin = []; preclin = []; recalllin = []; fscorelin = [];
% parfor i = 1:numofvalid/4
%     iter1 = startpoint_C*2^(4*(i-1));
%     iter2 = startpoint_C*2^(4*(i-1)+1);
%     iter3 = startpoint_C*2^(4*(i-1)+2);
%     iter4 = startpoint_C*2^(4*(i-1)+3);
%     [CCR1,prec1,recall1,fscore1] = svm_linearkernal(feature,label,numoffold,iter1);
%     [CCR2,prec2,recall2,fscore2] = svm_linearkernal(feature,label,numoffold,iter2);
%     [CCR3,prec3,recall3,fscore3] = svm_linearkernal(feature,label,numoffold,iter3);
%     [CCR4,prec4,recall4,fscore4] = svm_linearkernal(feature,label,numoffold,iter4);
%     CCRi= [CCR1 CCR2 CCR3 CCR4]; CCRlin = [CCRlin CCRi];
%     preci= [prec1 prec2 prec3 prec4]; preclin = [preclin preci];
%     recalli= [recall1 recall2 recall3 recall4]; recalllin = [recalllin recalli];
%     fscorei= [fscore1 fscore2 fscore3 fscore4]; fscorelin = [fscorelin fscorei];
% end
% % find best c
% [~,maxC_CCR_lin]=find(CCRlin==max(CCRlin(:)));
% [~,maxC_prec_lin]=find(preclin==max(preclin(:)));
% [~,maxC_rec_lin]=find(recalllin==max(recalllin(:)));
% [~,maxC_fsc_lin]=find(fscorelin==max(fscorelin(:)));
% % plotting
% fig = figure;
% sk = (log(startpoint_C)/log(2));
% plot(sk:sk-1+numofvalid,CCRlin,sk:sk-1+numofvalid,preclin,sk:sk-1+numofvalid,recalllin,...
%     sk:sk-1+numofvalid,fscorelin);
% legend('CCR','precession','recall','fscore');
% xlabel('2^C'); ylabel('preformance matrics');

% %% RBF kernal with k-fold
% CCRrbf = []; precrbf = []; recallrbf = []; fscorerbf = [];
% parfor i = 1:numofvalid/4
%     iter1 = startpoint_C*2^(4*(i-1));
%     iter2 = startpoint_C*2^(4*(i-1)+1);
%     iter3 = startpoint_C*2^(4*(i-1)+2);
%     iter4 = startpoint_C*2^(4*(i-1)+3);
%     [CCR1,prec1,recall1,fscore1] = svm_rbfkernal(feature,label,numoffold,iter1,startpoint_sigma,numofsigma);
%     [CCR2,prec2,recall2,fscore2] = svm_rbfkernal(feature,label,numoffold,iter2,startpoint_sigma,numofsigma);
%     [CCR3,prec3,recall3,fscore3] = svm_rbfkernal(feature,label,numoffold,iter3,startpoint_sigma,numofsigma);
%     [CCR4,prec4,recall4,fscore4] = svm_rbfkernal(feature,label,numoffold,iter4,startpoint_sigma,numofsigma);
%     CCRi= [CCR1' CCR2' CCR3' CCR4']; CCRrbf = [CCRrbf CCRi];
%     preci= [prec1' prec2' prec3' prec4']; precrbf = [precrbf preci];
%     recalli= [recall1' recall2' recall3' recall4']; recallrbf = [recallrbf recalli];
%     fscorei= [fscore1' fscore2' fscore3' fscore4']; fscorerbf = [fscorerbf fscorei];
% end
% % find c and sigma corresponding to max of each performance measure
% [~,maxC_CCR_sig]=find(CCRrbf==max(CCRrbf(:))); [~,maxsigma_CCR]=find(CCRrbf'==max(CCRrbf(:)));
% [~,maxC_prec_sig]=find(precrbf==max(precrbf(:))); [~,maxsigma_prec]=find(precrbf'==max(precrbf(:)));
% [~,maxC_rec_sig]=find(recallrbf==max(recallrbf(:))); [~,maxsigma_rec]=find(recallrbf'==max(recallrbf(:)));
% [~,maxC_fsc_sig]=find(fscorerbf==max(fscorerbf(:))); [~,maxsigma_fsc]=find(fscorerbf'==max(fscorerbf(:)));
% % plot heat map of each matrix
% precrbf(isnan(precrbf))=0; 
% recallrbf(isnan(recallrbf))=0; 
% fscorerbf(isnan(fscorerbf))=0;
% sc = (log(startpoint_C)/log(2));
% ss = (log(startpoint_sigma)/log(2));
% [X,Y] = meshgrid(sc:sc-1+numofvalid, ss:ss-1+numofsigma);
% subplot(2,2,1); contourf(X,Y,CCRrbf); colorbar; title('CCR');
% ylabel('2^sigma'); xlabel('2^C');
% subplot(2,2,2); contourf(X,Y,precrbf); colorbar; title('precision');
% ylabel('2^sigma'); xlabel('2^C');
% subplot(2,2,3); contourf(X,Y,recallrbf); colorbar; title('recall');
% ylabel('2^sigma'); xlabel('2^C');
% subplot(2,2,4); contourf(X,Y,fscorerbf); colorbar; title('f-score');
% ylabel('2^sigma'); xlabel('2^C');

%% Polynomial kernal with k-fold
CCRpoly = []; precpoly = []; recallpoly = []; fscorepoly = [];
parfor i = 1:numofvalid/4
    iter1 = startpoint_C*2^(4*(i-1));
    iter2 = startpoint_C*2^(4*(i-1)+1);
    iter3 = startpoint_C*2^(4*(i-1)+2);
    iter4 = startpoint_C*2^(4*(i-1)+3);
    [CCR1,prec1,recall1,fscore1] = svm_polykernal(feature,label,numoffold,iter1,numofpoly)
    [CCR2,prec2,recall2,fscore2] = svm_polykernal(feature,label,numoffold,iter2,numofpoly)
    [CCR3,prec3,recall3,fscore3] = svm_polykernal(feature,label,numoffold,iter3,numofpoly)
    [CCR4,prec4,recall4,fscore4] = svm_polykernal(feature,label,numoffold,iter4,numofpoly)
    CCRi= [CCR1' CCR2' CCR3' CCR4']; CCRpoly = [CCRpoly CCRi];
    preci= [prec1' prec2' prec3' prec4']; precpoly = [precpoly preci];
    recalli= [recall1' recall2' recall3' recall4']; recallpoly = [recallpoly recalli];
    fscorei= [fscore1' fscore2' fscore3' fscore4']; fscorepoly = [fscorepoly fscorei];
end
% find best c and poly
precpoly(isnan(precpoly))=0; 
recallpoly(isnan(recallpoly))=0; 
fscorepoly(isnan(fscorepoly))=0;
[~,maxC_CCR_poly]=find(CCRpoly==max(CCRpoly(:)));
[~,maxpoly_CCR]=find(CCRpoly'==max(CCRpoly(:)));
[~,maxC_prec_poly]=find(precpoly==max(precpoly(:)));
[~,maxpoly_prec]=find(precpoly'==max(precpoly(:)));
[~,maxC_rec_poly]=find(recallpoly==max(recallpoly(:)));
[~,maxpoly_rec]=find(recallpoly'==max(recallpoly(:)));
[~,maxC_fsc_poly]=find(fscorepoly==max(fscorepoly(:)));
[~,maxpoly_fsc]=find(fscorepoly'==max(fscorepoly(:)));
% plotting
fig = figure;
sk = (log(startpoint_C)/log(2));
[X,Y] = meshgrid(sk:sk-1+numofvalid, -10:numofpoly-11);
subplot(2,2,1); contourf(X,Y,CCRpoly); colorbar; title('CCR');
ylabel('2^ degree of poly'); xlabel('2^C');
subplot(2,2,2); contourf(X,Y,precpoly); colorbar; title('precision');
ylabel('2^ degree of poly'); xlabel('2^C');
subplot(2,2,3); contourf(X,Y,recallpoly); colorbar; title('recall');
ylabel('2^ degree of poly'); xlabel('2^C');
subplot(2,2,4); contourf(X,Y,fscorepoly); colorbar; title('f-score');
ylabel('2^ degree of poly'); xlabel('2^C');










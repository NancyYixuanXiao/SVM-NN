clear
% preprocessing csv to mat
[feature,label] = svm_csv2mat();

meanoffeat = mean(feature);
varoffeat = var(feature);
% fig1 = figure;
% subplot(2,1,1);plot(meanoffeat);
% subplot(2,1,2); plot(varoffeat);

[U,S,V] = svds(feature,15);
feat_new =U*S*V';

numoffold = 10; % number of k fold
numofvalid = 24; % number of cross validation for  c value linear kernal
startpoint_C = 2^(1); % which boxconstrain to start with
numofsigma = 24; % number of sigma to cross validate
startpoint_sigma = 2^(-1); % which sigma to start with

% CCRlin = []; preclin = []; recalllin = []; fscorelin = [];
% parfor i = 1:numofvalid/4
%     iter1 = startpoint_C*2^(4*(i-1));
%     iter2 = startpoint_C*2^(4*(i-1)+1);
%     iter3 = startpoint_C*2^(4*(i-1)+2);
%     iter4 = startpoint_C*2^(4*(i-1)+3);
%     [CCR1,prec1,recall1,fscore1] = svm_linearkernal(feat_new,label,numoffold,iter1);
%     [CCR2,prec2,recall2,fscore2] = svm_linearkernal(feat_new,label,numoffold,iter2);
%     [CCR3,prec3,recall3,fscore3] = svm_linearkernal(feat_new,label,numoffold,iter3);
%     [CCR4,prec4,recall4,fscore4] = svm_linearkernal(feat_new,label,numoffold,iter4);
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
% fig1 = figure;
% sk = (log(startpoint_C)/log(2)); hold on
% plot(sk:sk-1+numofvalid,CCRlin,sk:sk-1+numofvalid,preclin,sk:sk-1+numofvalid,recalllin,...
%     sk:sk-1+numofvalid,fscorelin);
% plot(maxC_CCR_lin-10,CCRlin(1,maxC_CCR_lin),'x','MarkerSize',20);
% legend('CCR','precession','recall','fscore');
% xlabel('C = 2^c'); ylabel('preformance matrics');
% txt1 = sprintf('best CCR, C = %d',2^(maxC_CCR_lin-10));
% text(1,CCRlin(1,maxC_CCR_lin),txt1);

CCRrbf = []; precrbf = []; recallrbf = []; fscorerbf = [];
parfor i = 1:numofvalid/4
    iter1 = startpoint_C*2^(4*(i-1));
    iter2 = startpoint_C*2^(4*(i-1)+1);
    iter3 = startpoint_C*2^(4*(i-1)+2);
    iter4 = startpoint_C*2^(4*(i-1)+3);
    [CCR1,prec1,recall1,fscore1] = svm_rbfkernal(feat_new,label,numoffold,iter1,startpoint_sigma,numofsigma);
    [CCR2,prec2,recall2,fscore2] = svm_rbfkernal(feat_new,label,numoffold,iter2,startpoint_sigma,numofsigma);
    [CCR3,prec3,recall3,fscore3] = svm_rbfkernal(feat_new,label,numoffold,iter3,startpoint_sigma,numofsigma);
    [CCR4,prec4,recall4,fscore4] = svm_rbfkernal(feat_new,label,numoffold,iter4,startpoint_sigma,numofsigma);
    CCRi= [CCR1' CCR2' CCR3' CCR4']; CCRrbf = [CCRrbf CCRi];
    preci= [prec1' prec2' prec3' prec4']; precrbf = [precrbf preci];
    recalli= [recall1' recall2' recall3' recall4']; recallrbf = [recallrbf recalli];
    fscorei= [fscore1' fscore2' fscore3' fscore4']; fscorerbf = [fscorerbf fscorei];
end
% find c and sigma corresponding to max of each performance measure
[~,maxC_CCR_sig]=find(CCRrbf==max(CCRrbf(:))); [~,maxsigma_CCR]=find(CCRrbf'==max(CCRrbf(:)));
[~,maxC_prec_sig]=find(precrbf==max(precrbf(:))); [~,maxsigma_prec]=find(precrbf'==max(precrbf(:)));
[~,maxC_rec_sig]=find(recallrbf==max(recallrbf(:))); [~,maxsigma_rec]=find(recallrbf'==max(recallrbf(:)));
[~,maxC_fsc_sig]=find(fscorerbf==max(fscorerbf(:))); [~,maxsigma_fsc]=find(fscorerbf'==max(fscorerbf(:)));
% plot heat map of each matrix
fig2 = figure;
precrbf(isnan(precrbf))=0; 
recallrbf(isnan(recallrbf))=0; 
fscorerbf(isnan(fscorerbf))=0;
sc = (log(startpoint_C)/log(2));
ss = (log(startpoint_sigma)/log(2));
[X,Y] = meshgrid(sc:sc-1+numofvalid, ss:ss-1+numofsigma);hold on
subplot(2,2,1); contourf(X,Y,CCRrbf); colorbar; title('CCR');
ylabel('Sigma = 2^s'); xlabel('C = 2^c'); hold on
plot(maxC_CCR_sig,maxsigma_CCR-2,'x','MarkerSize',20); hold off
subplot(2,2,2); contourf(X,Y,precrbf); colorbar; title('precision');hold on
ylabel('Sigma = 2^s'); xlabel('C = 2^c');hold on
[~,mm]=find(precrbf(max(maxC_prec_sig),:)==max(precrbf(max(maxC_prec_sig),:)));
subplot(2,2,3); contourf(X,Y,recallrbf); colorbar; title('recall');hold on
ylabel('Sigma = 2^s'); xlabel('C = 2^c'); hold on
[mk,~]=find(recallrbf(:,max(maxC_rec_sig))==max(recallrbf(:,max(maxC_rec_sig))));
subplot(2,2,4); contourf(X,Y,fscorerbf); colorbar; title('f-score');hold on
ylabel('Sigma = 2^s'); xlabel('C = 2^c'); hold on
plot(maxC_fsc_sig,maxsigma_fsc-2,'x','MarkerSize',20); hold off




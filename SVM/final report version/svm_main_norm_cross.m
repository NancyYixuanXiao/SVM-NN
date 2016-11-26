clear
% preprocessing csv to mat
[feature,label] = svm_csv2mat();

%% enter parameters
% which kernel model you want to run
% 1 is linear, 2 is RBF and 3 is polynomial
model = 2;

% which regulization method you want to use
% 1 is L1 morm, 2 is L2 norm
norm = 2;

% number of k fold
numoffold = 10; 

% parameters for ALL kernal SVM
numofvalid = 16; % number of cross validation for C value 

which_C_to_start = -6; % 2^(which_C_to_start), minimum is -9
startpoint_C = 2^(which_C_to_start); % which boxconstrain to start with

% parameters for RBF kernal SVM
numofsigma = 16; % number of sigma to cross validate
which_Sigma_to_start = -1; % 2^(which_Sigma_to_start)
startpoint_sigma = 2^(which_Sigma_to_start); % which sigma to start with

% parameters for polynomial kernal SVM
numofpoly = 4; % start at 2^(1), number of polynomial degree to cross validate

if norm ==1
    norm_method = 'SMO';
elseif norm == 2
        norm_method = 'QP';
else 
    disp('please choose between L1 and L2 norm');
    return
end

if which_C_to_start < -9
    disp('C must be greater than 2^-9')
    return
end

if numofvalid/4-round(numofvalid/4) ~= 0
    disp('number of validation must be factor of 4');
    return
end

%% linear kernal with k-fold
if model ==1
    CCRlin = []; preclin = []; recalllin = []; fscorelin = [];
    parfor i = 1:numofvalid/4
        % C value for each parallel worker
        iter1 = startpoint_C*2^(4*(i-1)); iter2 = startpoint_C*2^(4*(i-1)+1);
        iter3 = startpoint_C*2^(4*(i-1)+2); iter4 = startpoint_C*2^(4*(i-1)+3);
        % parallel cross validating on C
        [CCR1,prec1,recall1,fscore1] = svm_linearkernal(norm_method,feature,label,...
            numoffold,iter1);
        [CCR2,prec2,recall2,fscore2] = svm_linearkernal(norm_method,feature,label,...
        numoffold,iter2);
        [CCR3,prec3,recall3,fscore3] = svm_linearkernal(norm_method,feature,label,...
        numoffold,iter3);
        [CCR4,prec4,recall4,fscore4] = svm_linearkernal(norm_method,feature,label,...
        numoffold,iter4);
        % record result on CCR, precision, recall and f-score
        CCRi= [CCR1 CCR2 CCR3 CCR4]; CCRlin = [CCRlin CCRi];
        preci= [prec1 prec2 prec3 prec4]; preclin = [preclin preci];
        recalli= [recall1 recall2 recall3 recall4]; recalllin = [recalllin recalli];
        fscorei= [fscore1 fscore2 fscore3 fscore4]; fscorelin = [fscorelin fscorei];
    end
    
    % find best C for optimal CCR, precision, recall and f-score
    [~,maxC_CCR_lin]=find(CCRlin==max(CCRlin(:)));
    [~,maxC_prec_lin]=find(preclin==max(preclin(:)));
    [~,maxC_rec_lin]=find(recalllin==max(recalllin(:)));
    [~,maxC_fsc_lin]=find(fscorelin==max(fscorelin(:)));
    
    % plotting
    fig1 = figure;
    sk = (log(startpoint_C)/log(2)); hold on % shift ploting start point
    plot(sk:sk-1+numofvalid,CCRlin,sk:sk-1+numofvalid,preclin,sk:sk-1+numofvalid,...
        recalllin,sk:sk-1+numofvalid,fscorelin);
    plot(maxC_CCR_lin+which_C_to_start-1,CCRlin(1,maxC_CCR_lin),...
        'x','MarkerSize',40); % mark optimal CCR
    legend('CCR','precession','recall','fscore');
    xlabel('C = 2^c'); ylabel('preformance matrics');
    txt1 = sprintf('best CCR, C = %d',2^(maxC_CCR_lin-10));
    text(1,CCRlin(1,maxC_CCR_lin),txt1)
end

%% RBF kernal with k-fold
if model == 2
    CCRrbf = []; precrbf = []; recallrbf = []; fscorerbf = [];
    parfor i = 1:numofvalid/4
        % C value for each parallel worker
        iter1 = startpoint_C*2^(4*(i-1)); iter2 = startpoint_C*2^(4*(i-1)+1);
        iter3 = startpoint_C*2^(4*(i-1)+2); iter4 = startpoint_C*2^(4*(i-1)+3);
        % parallel cross validating on C, Sigma validation is in the
        % function itself
        [CCR1,prec1,recall1,fscore1] = svm_rbfkernal(norm_method,feature,label,numoffold,...
            iter1,startpoint_sigma,numofsigma);
        [CCR2,prec2,recall2,fscore2] = svm_rbfkernal(norm_method,feature,label,numoffold,...
            iter2,startpoint_sigma,numofsigma);
        [CCR3,prec3,recall3,fscore3] = svm_rbfkernal(norm_method,feature,label,numoffold,...
            iter3,startpoint_sigma,numofsigma);
        [CCR4,prec4,recall4,fscore4] = svm_rbfkernal(norm_method,feature,label,numoffold,...
            iter4,startpoint_sigma,numofsigma);
        % record result on CCR, precision, recall and f-score
        CCRi= [CCR1' CCR2' CCR3' CCR4']; CCRrbf = [CCRrbf CCRi];
        preci= [prec1' prec2' prec3' prec4']; precrbf = [precrbf preci];
        recalli= [recall1' recall2' recall3' recall4']; recallrbf = [recallrbf recalli];
        fscorei= [fscore1' fscore2' fscore3' fscore4']; fscorerbf = [fscorerbf fscorei];
    end
    
    % find best C AND Sigma for optimal CCR, precision, recall and f-score
    [~,maxC_CCR_sig]=find(CCRrbf==max(CCRrbf(:))); 
    [~,maxsigma_CCR]=find(CCRrbf'==max(CCRrbf(:)));
    [~,maxC_prec_sig]=find(precrbf==max(precrbf(:))); 
    [~,maxsigma_prec]=find(precrbf'==max(precrbf(:)));
    [~,maxC_rec_sig]=find(recallrbf==max(recallrbf(:))); 
    [~,maxsigma_rec]=find(recallrbf'==max(recallrbf(:)));
    [~,maxC_fsc_sig]=find(fscorerbf==max(fscorerbf(:))); 
    [~,maxsigma_fsc]=find(fscorerbf'==max(fscorerbf(:)));
    
    % plot heat map of CCR, precision, recall and f-score
    fig2 = figure;
    precrbf(isnan(precrbf))=0; recallrbf(isnan(recallrbf))=0; 
    fscorerbf(isnan(fscorerbf))=0; % remove NAN due to misprediction 
    sc = (log(startpoint_C)/log(2));
    ss = (log(startpoint_sigma)/log(2)); % shift start point
    [X,Y] = meshgrid(sc:sc-1+numofvalid, ss:ss-1+numofsigma);
    
    subplot(2,2,1); hold on; 
    contourf(X,Y,CCRrbf); colorbar; title('CCR');
    ylabel('Sigma = 2^s'); xlabel('C = 2^c');
%     plot(maxC_CCR_sig+which_C_to_start-1,maxsigma_CCR+which_Sigma_to_start-1,...
%         'x','MarkerSize',20); hold off
    
    subplot(2,2,2); hold on; 
    contourf(X,Y,precrbf); colorbar; title('precision');
    ylabel('Sigma = 2^s'); xlabel('C = 2^c');
    [~,mm]=find(precrbf(max(maxC_prec_sig),:)==max(precrbf(max(maxC_prec_sig),:)));
%     plot(max(maxC_prec_sig)+which_C_to_start-1,max(mm)+which_Sigma_to_start-1,...
%         'x','MarkerSize',20); hold off
    
    subplot(2,2,3); hold on;
    contourf(X,Y,recallrbf); colorbar; title('recall');
    ylabel('Sigma = 2^s'); xlabel('C = 2^c');
    [mk,~]=find(recallrbf(:,max(maxC_rec_sig))==max(recallrbf(:,max(maxC_rec_sig))));
%     plot(max(maxC_rec_sig)+which_C_to_start-1,min(mk)+which_Sigma_to_start-1,...
%         'x','MarkerSize',20); hold off
    
    subplot(2,2,4); hold on
    contourf(X,Y,fscorerbf); colorbar; title('f-score');
    ylabel('Sigma = 2^s'); xlabel('C = 2^c');
%     plot(maxC_fsc_sig+which_C_to_start-1,maxsigma_fsc+which_Sigma_to_start-1,...
%         'x','MarkerSize',20); hold off
end

%% Polynomial kernal with k-fold
if model == 3
    CCRpoly = []; precpoly = []; recallpoly = []; fscorepoly = [];
    parfor i = 1:numofvalid/4
        % C value for each parallel worker
        iter1 = startpoint_C*2^(4*(i-1)); iter2 = startpoint_C*2^(4*(i-1)+1);
        iter3 = startpoint_C*2^(4*(i-1)+2); iter4 = startpoint_C*2^(4*(i-1)+3);
        % parallel cross validating on C, poly degree validation is in the
        % function itself
        [CCR1,prec1,recall1,fscore1] = svm_polykernal(norm_method,feature,label,numoffold,...
            iter1,numofpoly)
        [CCR2,prec2,recall2,fscore2] = svm_polykernal(norm_method,feature,label,numoffold,...
            iter2,numofpoly)
        [CCR3,prec3,recall3,fscore3] = svm_polykernal(norm_method,feature,label,numoffold,...
            iter3,numofpoly)
        [CCR4,prec4,recall4,fscore4] = svm_polykernal(norm_method,feature,label,numoffold,...
            iter4,numofpoly)
        CCRi= [CCR1' CCR2' CCR3' CCR4']; CCRpoly = [CCRpoly CCRi];
        preci= [prec1' prec2' prec3' prec4']; precpoly = [precpoly preci];
        recalli= [recall1' recall2' recall3' recall4']; recallpoly = [recallpoly recalli];
        fscorei= [fscore1' fscore2' fscore3' fscore4']; fscorepoly = [fscorepoly fscorei];
    end
    
    % find best C AND Degree for optimal CCR, precision, recall and f-score
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
end




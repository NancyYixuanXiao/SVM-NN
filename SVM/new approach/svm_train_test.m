function [CCR,recall] = svm_train_test(featrd,kfold,kernel,norm_method,...
    sigma_iter,c_iter,feature,label) 
warning('off','all')
c1 = cvpartition(label,'KFold',kfold); 

%% rbf kernel
if strcmp(kernel,'rbf') == 1
    CCR = zeros(length(sigma_iter),length(c_iter));
    CCRi=zeros(1,kfold);
%     precision = zeros(length(sigma_iter),length(c_iter));
%     precisioni=zeros(1,kfold);
    recall = zeros(length(sigma_iter),length(c_iter));
    recalli=zeros(1,kfold);
%     fscore = zeros(length(sigma_iter),length(c_iter));
%     fscorei=zeros(1,kfold);
    for i = 1:length(sigma_iter)
        sigma = sigma_iter(1,i);
        
        for k = 1:length(c_iter)
            C = c_iter(1,k);
            
            for j = 1:kfold
                trIdx = c1.training(j);
                feature = bsxfun(@minus,feature,mean(feature));
                xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
                xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
                if featrd == 1
                [V,~]=eig(xtrain'*xtrain);
                xtrain=xtrain*V(:,1:15); xtest=xtest*V(:,1:15); % change num of feature you wanna use
%                 coef = pca(xtrain); 
%                 xtrain = xtrain*coef(:,1:3); xtest = xtest*coef(:,1:3);
                end
                svmpara=svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
                    C,'kernel_function',kernel,'tolkkt',2^-14,'options',...
                    statset('Display','off','MaxIter',10^7),'rbf_sigma',sigma,...
                    'kernelcachelimit',10000000,'method',norm_method);
                result=svmclassify(svmpara,xtest);
                
                conf=confusionmat(testlabel,result);
%                 precisioni(1,j)=conf(1,1)/(conf(1,1)+conf(2,1));
                recalli(1,j)=conf(1,1)/(conf(1,1)+conf(1,2));
%                 fscorei(1,j)=2*preci(1,j)*recalli(1,j)/(preci(1,j)+recalli(1,j));
                CCRi(1,j)=length(find((testlabel-result)==0))/length(testlabel);
            end
            CCR(i,k) = mean(CCRi);
%             precision(i,k)=mean(precisioni);
            recall(i,k)=mean(recalli);
%             fscore(i,k)=mean(fscorei);
        end
        i
    end    
end

%% polynomial kernel
% if strcmp(kernel,'polynomial') == 1
%     CCR = zeros(length(pdeg_iter),length(c_iter));
%     CCRi=zeros(1,kfold);
%     for i = 1:length(pdeg_iter)
%         pdeg = pdeg_iter(1,i);
%         
%         for k = 1:length(c_iter)
%             C = c_iter(1,k);
%             
%             for j = 1:kfold
%                 trIdx = c1.training(j);
%                 feature = bsxfun(@minus,feature,mean(feature));
%                 xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
%                 xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
%                 if featrd == 1
%                     coef = pca(xtrain); 
%                     xtrain = xtrain*coef(:,1:3); xtest = xtest*coef(:,1:3);
%                 end
%                 svmpara=svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
%                     C,'kernel_function',kernel,'options',...
%                     statset('Display','off','MaxIter',10^8),'polyorder',pdeg,...
%                     'kernelcachelimit',10000000,'method',norm_method);
%                 result=svmclassify(svmpara,xtest);
%                 
%                 CCRi(1,j)=length(find((testlabel-result)==0))/length(testlabel);
%             end
%             CCR(i,k) = mean(CCRi);
%         end
%     end    
% end

%% linear kernel
if strcmp(kernel,'linear') == 1
    CCRi=zeros(1,kfold); CCR = zeros(1,length(c_iter));
    precision = zeros(1,length(c_iter));
    precisioni=zeros(1,kfold);
    recall = zeros(1,length(c_iter));
    recalli=zeros(1,kfold);
    fscore = zeros(1,length(c_iter));
    fscorei=zeros(1,kfold);
    for k = 1:length(c_iter)
        C = c_iter(1,k);
        for j = 1:kfold
            trIdx = c1.training(j);
            feature = bsxfun(@minus,feature,mean(feature));
            xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
            xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
            if featrd == 1
                [V,~]=eig(xtrain'*xtrain);
                xtrain=xtrain*V(:,1:25); xtest=xtest*V(:,1:25); % change num of feature you wanna use
%                 coef = pca(xtrain); 
%                 xtrain = xtrain*coef(:,1:23); xtest = xtest*coef(:,1:23);
            end
%             if strcmp(norm_method,'QP') == 1
%             svmpara=svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
%                 C,'kernel_function',kernel,'tolkkt',2^-6,'options',...
%                 statset('Display','off','MaxIter',10^7),...
%                 'kernelcachelimit',1000000,'method',norm_method);
%             result=svmclassify(svmpara,xtest);
%             else
                svmpara=fitcsvm(xtrain,trainlabel,'Cost',[0,1;1,0],'CacheSize',...
                    1000000,'NumPrint',150000,'BoxConstraint',C);
                result=predict(svmpara,xtest);
%             end
            
            conf=confusionmat(testlabel,result);
%             precisioni(1,j)=conf(1,1)/(conf(1,1)+conf(2,1));
            recalli(1,j)=conf(1,1)/(conf(1,1)+conf(1,2));
%             fscorei(1,j)=2*preci(1,j)*recalli(1,j)/(preci(1,j)+recalli(1,j));
            CCRi(1,j)=length(find((testlabel-result)==0))/length(testlabel);
        end
        CCR(1,k) = mean(CCRi);
%         precision(1,k)=mean(precisioni);
        recall(1,k)=mean(recalli);
%         fscore(1,k)=mean(fscorei);
    end
    k
end

end
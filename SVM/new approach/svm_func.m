function [CCR] = svm_func(kfold,kernel,norm_method,pdeg_iter,sigma_iter,c_iter,FNcost...
    ,feature,label) 
% warning('off','all')
c1 = cvpartition(label,'KFold',kfold); 

if strcmp(kernel,'rbf') == 1
    CCR = zeros(length(sigma_iter),length(c_iter));
    CCRi=zeros(1,kfold);
    for i = 1:length(sigma_iter)
        sigma = sigma_iter(1,i);
        
        for k = 1:length(c_iter)
            C = c_iter(1,k);
            
            for j = 1:kfold
                trIdx = c1.training(j);
                xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
                xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
                svmpara=svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
                    C,'kernel_function',kernel,'tolkkt',2^-14,'options',...
                    statset('Display','off','MaxIter',10^7),'rbf_sigma',sigma,...
                    'kernelcachelimit',10000000,'method',norm_method);
                result=svmclassify(svmpara,xtest);
                
                CCRi(1,j)=length(find((testlabel-result)==0))/length(testlabel);
            end
            CCR(i,k) = mean(CCRi);
        end
    end    
end

if strcmp(kernel,'polynomial') == 1
    CCR = zeros(length(pdeg_iter),length(c_iter));
    CCRi=zeros(1,kfold);
    for i = 1:length(pdeg_iter)
        pdeg = pdeg_iter(1,i);
        
        for k = 1:length(c_iter)
            C = c_iter(1,k);
            
            for j = 1:kfold
                trIdx = c1.training(j);
                xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
                xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
                svmpara=svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
                    C,'kernel_function',kernel,'options',...
                    statset('Display','off','MaxIter',10^8),'polyorder',pdeg,...
                    'kernelcachelimit',10000000,'method',norm_method);
                result=svmclassify(svmpara,xtest);
                
                CCRi(1,j)=length(find((testlabel-result)==0))/length(testlabel);
            end
            CCR(i,k) = mean(CCRi);
        end
    end    
end

if strcmp(kernel,'linear') == 1
    CCRi=zeros(1,kfold); CCR = zeros(1,length(c_iter));
    c1 = cvpartition(label,'KFold',kfold);
    for k = 1:length(c_iter)
        C = c_iter(1,k);
        for j = 1:kfold
            trIdx = c1.training(j);
            xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
            xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
            if strcmp(norm_method,'QP') == 1
                svmpara=svmtrain(xtrain,trainlabel,'autoscale',false,'boxconstrain',...
                    C,'kernel_function',kernel,'tolkkt',2^-6,'options',...
                    statset('Display','off','MaxIter',10^7),...
                    'kernelcachelimit',1000000,'method',norm_method);
                result=svmclassify(svmpara,xtest);
            else
                svmpara=fitcsvm(xtrain,trainlabel,'Cost',[0,1;FNcost,0],'CacheSize',...
                    1000000,'NumPrint',150000,'BoxConstraint',C);
                result=predict(svmpara,xtest);
            end
            
            CCRi(1,j)=length(find((testlabel-result)==0))/length(testlabel);
        end
        CCR(1,k) = mean(CCRi);
    end
end

end
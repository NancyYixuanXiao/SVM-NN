function [CCR,xtrain,trainlabel,xtest,result] = svm_pred(kernel,norm_method,sigma,C,feature,label)


% warning('off','all')
c1 = cvpartition(label,'KFold',3); 

if strcmp(kernel,'rbf') == 1
    trIdx = c1.training(1);
    feature = bsxfun(@minus,feature,mean(feature));
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
    coef = pca(xtrain);
    xtrain = xtrain*coef(:,1:3); xtest = xtest*coef(:,1:3);
    svmStruct=svmtrain(xtrain,trainlabel,'showplot','false','kernel_function',kernel,...
      'boxconstraint',C,'kktviolationlevel',0.05,'tolkkt',5e-3,'method',norm_method...
      ,'rbf_sigma',sigma);
    result=svmclassify(svmStruct,xtest);
    plot3(xtest(result==2,1),xtest(result==2,2),xtest(result==2,3),'r.','MarkerSize',12); hold on
    plot3(xtest(result==1,1),xtest(result==1,2),xtest(result==1,3),'b.','MarkerSize',12);
    
    CCR=length(find((testlabel-result)==0))/length(testlabel);
end

if strcmp(kernel,'linear') == 1
    trIdx = c1.training(1);
    feature = bsxfun(@minus,feature,mean(feature));
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);
    coef = pca(xtrain);
    xtrain = xtrain*coef(:,1:3); xtest = xtest*coef(:,1:3);
    svmStruct=svmtrain(xtrain,trainlabel,'showplot','false','kernel_function',kernel,...
      'boxconstraint',C,'kktviolationlevel',0.05,'tolkkt',5e-3,'method',norm_method...
      ,'rbf_sigma',sigma);
    result=svmclassify(svmStruct,xtest);
    
    plot3(xtest(result==2,1),xtest(result==2,2),xtest(result==2,3),'r.','MarkerSize',12); hold on
    plot3(xtest(result==1,1),xtest(result==1,2),xtest(result==1,3),'b.','MarkerSize',12);
    
    CCR=length(find((testlabel-result)==0))/length(testlabel);
end






%% ploting
sv =  svmStruct.SupportVectors;
alphaHat = svmStruct.Alpha;
bias = svmStruct.Bias;
kfun = svmStruct.KernelFunction;
kfunargs = svmStruct.KernelFunctionArgs;
sh = svmStruct.ScaleData.shift; % shift vector
scalef = svmStruct.ScaleData.scaleFactor; % scale vector

% scale and shift data
k = 50; 
cubeXMin = min(xtrain1(:,1));
cubeYMin = min(xtrain1(:,2));
cubeZMin = min(xtrain1(:,3));

cubeXMax = max(xtrain1(:,1));
cubeYMax = max(xtrain1(:,2));
cubeZMax = max(xtrain1(:,3));
stepx = (cubeXMax-cubeXMin)/(k-1);
stepy = (cubeYMax-cubeYMin)/(k-1);
stepz = (cubeZMax-cubeZMin)/(k-1);
[x, y, z] = meshgrid(cubeXMin:stepx:cubeXMax,cubeYMin:stepy:cubeYMax,cubeZMin:stepz:cubeZMax);
mm = size(x);
x = x(:);
y = y(:);
z = z(:);
f = (feval(kfun,sv,[x y z],kfunargs{:})'*alphaHat(:)) + bias;

% unscale and unshift data 
x =(x./repmat(scalef(1),size(x,1),1)) - repmat(sh(1),size(x,1),1);
y =(y./repmat(scalef(2),size(y,1),1)) - repmat(sh(2),size(y,1),1);
z =(z./repmat(scalef(3),size(z,1),1)) - repmat(sh(3),size(z,1),1);

x0 = reshape(x, mm);
y0 = reshape(y, mm);
z0 = reshape(z, mm);
v0 = reshape(f, mm);

[faces,verts,~] = isosurface(x0, y0, z0, v0, 0, x0);
patch('Vertices', verts, 'Faces', faces, 'FaceColor','k','edgecolor', 'none', 'FaceAlpha', 0.5);
grid on
box on
view(3)
hold off
end
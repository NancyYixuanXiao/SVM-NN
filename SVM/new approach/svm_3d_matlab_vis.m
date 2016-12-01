function [CCR] = svm_3d_matlab_vis(U,label)
fig1 = figure;
plot3(U(label==2,1),U(label==2,2),U(label==2,3),'r.','MarkerSize',12); hold on
plot3(U(label==1,1),U(label==1,2),U(label==1,3),'b.','MarkerSize',12);


c1 = cvpartition(label,'KFold',3); % k folding
trIdx = c1.training(1);
xtrain=U(trIdx==1,:); trainlabel=label(trIdx==1,:);
xtest=U(trIdx==0,:); testlabel=label(trIdx==0,:);
svmStruct = svmtrain(xtrain,trainlabel,'showplot','false','kernel_function','rbf',...
      'boxconstraint',1024,'kktviolationlevel',0.05,'tolkkt',5e-3,'method','QP','rbf_sigma',0.5);
result=svmclassify(svmStruct,xtest);
CCR=length(find((testlabel-result)==0))/length(testlabel);

trainlabel = num2cell(trainlabel);

sv =  svmStruct.SupportVectors;
alphaHat = svmStruct.Alpha;
bias = svmStruct.Bias;
kfun = svmStruct.KernelFunction;
kfunargs = svmStruct.KernelFunctionArgs;
sh = svmStruct.ScaleData.shift; % shift vector
scalef = svmStruct.ScaleData.scaleFactor; % scale vector

trainlabel = trainlabel(~any(isnan(xtrain),2));
xtrain =xtrain(~any(isnan(xtrain),2),:); % remove rows with NaN 

% scale and shift data
xtrain1 = repmat(scalef,size(xtrain,1),1).*(xtrain+repmat(sh,size(xtrain,1),1));
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
t = strcmp(trainlabel, trainlabel{1});

% unscale and unshift data 
xtrain1 =(xtrain1./repmat(scalef,size(xtrain,1),1)) - repmat(sh,size(xtrain,1),1);
x =(x./repmat(scalef(1),size(x,1),1)) - repmat(sh(1),size(x,1),1);
y =(y./repmat(scalef(2),size(y,1),1)) - repmat(sh(2),size(y,1),1);
z =(z./repmat(scalef(3),size(z,1),1)) - repmat(sh(3),size(z,1),1);


% % load unscaled support vectors for plotting
% sv = svmStruct.SupportVectorIndices;
% sv = [xtrain1(sv, :)];
% plot3(sv(:, 1), sv(:, 2), sv(:, 3), 'go');

x0 = reshape(x, mm);
y0 = reshape(y, mm);
z0 = reshape(z, mm);
v0 = reshape(f, mm);

[faces,verts,colors] = isosurface(x0, y0, z0, v0, 0, x0);
patch('Vertices', verts, 'Faces', faces, 'FaceColor','k','edgecolor', 'none', 'FaceAlpha', 0.5);
grid on
box on
view(3)
hold off
end
function svm_3d_vis(svmStruct,xtest,testlabel,psv)
%% ploting

testlabel = num2cell(testlabel);
sv = svmStruct.SupportVectors;

alphaHat = svmStruct.Alpha;
bias = svmStruct.Bias;
kfun = svmStruct.KernelFunction;
kfunargs = svmStruct.KernelFunctionArgs;
sh = svmStruct.ScaleData.shift; % shift vector
scalef = svmStruct.ScaleData.scaleFactor; % scale vector

xtest =xtest(~any(isnan(xtest),2),:); % remove rows with NaN 

% scale and shift data
xtest1 = repmat(scalef,size(xtest,1),1).*(xtest+repmat(sh,size(xtest,1),1));
k = 50; 
cubeXMin = min(xtest1(:,1));
cubeYMin = min(xtest1(:,2));
cubeZMin = min(xtest1(:,3));

cubeXMax = max(xtest1(:,1));
cubeYMax = max(xtest1(:,2));
cubeZMax = max(xtest1(:,3));
stepx = (cubeXMax-cubeXMin)/(k-1);
stepy = (cubeYMax-cubeYMin)/(k-1);
stepz = (cubeZMax-cubeZMin)/(k-1);
[x, y, z] = meshgrid(cubeXMin:stepx:cubeXMax,cubeYMin:stepy:cubeYMax,cubeZMin:stepz:cubeZMax);
mm = size(x);
x = x(:);
y = y(:);
z = z(:);
f = (feval(kfun,sv,[x y z],kfunargs{:})'*alphaHat(:)) + bias;

if psv==1
    sv = svmStruct.SupportVectorIndices;
    sv = xtest(sv, :);
    plot3(sv(:, 1), sv(:, 2), sv(:, 3), 'go');
end

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
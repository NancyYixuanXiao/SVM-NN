function svm_NN()
c1 = cvpartition(label,'KFold',10); % k folding
trIdx = c1.training(1);
    xtrain=feature(trIdx==1,:); trainlabel=label(trIdx==1,:);
    xtest=feature(trIdx==0,:); testlabel=label(trIdx==0,:);


net = network( ...
2, ... % numInputs, number of inputs,
2, ... % numLayers, number of layers
[1  ; 0], ... % biasConnect, numLayers-by-1 Boolean vector,
[ones(1,2); zeros(1,2)], ... % inputConnect, numLayers-by-numInputs Boolean matrix,
[0 0; 1 0], ... % layerConnect, numLayers-by-numLayers Boolean matrix
[0 1] ... % outputConnect, 1-by-numLayers Boolean vector
);

net.layers{1}.size = 5;
% hidden layer transfer function
net.layers{1}.transferFcn = 'logsig';
tratmp = [xtrain(:,1);xtrain(:,2)];
testmp = [xtest(:,1);xtest(:,2)];

net = configure(net,tratmp',[trainlabel;trainlabel]');

initial_output = net(tratmp');
% network training
net.trainFcn = 'trainlm';
net.performFcn = 'mse';
net = train(net,tratmp',trainlabel');
% network response after training
tmp = net(testmp');
final_output = round(tmp');
tmp = find((testlabel-final_output)==0);
end
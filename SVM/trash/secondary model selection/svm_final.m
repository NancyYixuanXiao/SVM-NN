clear
% repeated validation
iter = 40;

CCR_sum = zeros(11,8,iter); 
for i = 1:iter
    [CCR_iter] = svm_func_2_final();
    CCR_sum(:,:,i) = CCR_iter;
    i
end
% plot the average result 
max_iter = zeros(4,iter); tmp = zeros(6,1); svd_val = zeros(iter,1);
for i = 1:iter
    for j = 1:2
        max_iter(j,i) = max(CCR_sum(j,:,i));
    end
    for j = 3:5
        tmp(j-2,1) = max(CCR_sum(j,:,i));
    end
    max_iter(3,i) = max(tmp);
    for j = 6:11
        tmp(j-5,1) = max(CCR_sum(j,:,i));
    end
    [max_iter(4,i),svd_val(i,1)] = max(tmp);
end
average = mean(max_iter,2);
minimum = min(max_iter,[],2);
maximum = max(max_iter,[],2);

fig1 = figure;
plot(max_iter'); title('Best CCR from each iteration');
xlabel('iterations'); ylabel('CCR');
legend('L1 norm linear kernel','L2 norm linear kernel',...
    'L2 norm rbf kernel','L2 norm linear kernel with SVD');

fig2 = figure;
subplot(3,1,1);
plot(average,'-o'); title('mean of best CCR of each model');
xlabel('models'); ylabel('CCR');
text(1,average(1,1),'L1 norm linear kernel')
text(2,average(2,1),'L2 norm linear kernel')
text(3,average(3,1),'L2 norm rbf kernel')
text(3,average(4,1)+0.001,'L2 norm linear kernel with SVD')
subplot(3,1,2);
plot(maximum,'-o'); title('max of best CCR of each model');
xlabel('models'); ylabel('CCR');
text(1,maximum(1,1),'L1 norm linear kernel')
text(2,maximum(2,1),'L2 norm linear kernel')
text(3,maximum(3,1),'L2 norm rbf kernel')
text(3,maximum(4,1)-0.001,'L2 norm linear kernel with SVD')
subplot(3,1,3);
plot(minimum,'-o'); title('min of best CCR of each model');
xlabel('models'); ylabel('CCR');
text(1,minimum(1,1),'L1 norm linear kernel')
text(2,minimum(2,1),'L2 norm linear kernel')
text(3,minimum(3,1),'L2 norm rbf kernel')
text(3,minimum(4,1)+0.001,'L2 norm linear kernel with SVD')

fig3 = figure;
C = categorical(svd_val,[1 2 3 4 5 6],{'28','26','24','22','20','18'});
histogram(C,'BarWidth',0.5);

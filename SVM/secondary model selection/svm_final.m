clear
% repeated validation
iter = 20;

CCR_sum = zeros(8,8,iter); 
for i = 1:iter
    [CCR_iter] = svm_func_2_final();
    CCR_sum(:,:,i) = CCR_iter;
    i
end
% plot the average result 
max_iter = zeros(4,iter); tmp = zeros(3,1);
for i = 1:iter
    for j = 1:2
        max_iter(j,i) = max(CCR_sum(j,:,i));
    end
    for j = 3:5
        tmp(j-2,1) = max(CCR_sum(j,:,i));
    end
    max_iter(3,i) = max(tmp);
    for j = 6:8
        tmp(j-5,1) = max(CCR_sum(j,:,i));
    end
    max_iter(4,i) = max(tmp);
end
average = mean(max_iter,2);
minimum = min(max_iter,[],2);
maximum = max(max_iter,[],2);

fig1 = figure;
plot(max_iter); title('Best CCR from each iteration');
xlabel('iterations'); ylabel('CCR');

fig2 = figure;
subplot(3,1,1);
plot(average,'-o'); title('mean of best CCR of each model');
xlabel('models'); ylabel('CCR');
subplot(3,1,2);
plot(minimum,'-o'); title('min of best CCR of each model');
xlabel('models'); ylabel('CCR');
subplot(3,1,3);
plot(maximum,'-o'); title('max of best CCR of each model');
xlabel('models'); ylabel('CCR');



function [feature,label] = svm_csv2mat()
data = csv2cell('data.csv','fromfile');

mcount = 0;
for i = 2:569
    if data{i,2}=='M'
        mcount = mcount+1;
    end
end

data_m = zeros(mcount,30);
data_b = zeros(569-mcount,30);
countm = 1; countb = 1;
for i = 2:569
    if data{i,2}=='M'
        for j = 1:30
            data_m(countm,j) = str2double(cell2mat(data(i,j+2)));
        end
        countm = countm+1;
    else
        for j = 1:30
            data_b(countb,j) = str2double(cell2mat(data(i,j+2)));
        end
        countb = countb+1;
    end
end
feature = [data_m;data_b];
label = [ones(mcount,1);2*ones(569-mcount,1)];
end
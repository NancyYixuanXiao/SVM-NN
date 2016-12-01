function [feature_wdbc,label_wdbc,feature_wpdc,label_wpdc] = svm_cell2mat()
% wbdc
data_wdbc = csv2cell('wdbc.csv','fromfile');
mcount = 0;
for i = 2:569
    if data_wdbc{i,2}=='M'
        mcount = mcount+1;
    end
end

data_m = zeros(mcount,30);
data_b = zeros(569-mcount,30);
countm = 1; countb = 1;
for i = 2:569
    if data_wdbc{i,2}=='M'
        for j = 1:30
            data_m(countm,j) = str2double(cell2mat(data_wdbc(i,j+2)));
        end
        countm = countm+1;
    else
        for j = 1:30
            data_b(countb,j) = str2double(cell2mat(data_wdbc(i,j+2)));
        end
        countb = countb+1;
    end
end
feature_wdbc = [data_m;data_b];
label_wdbc = [ones(mcount,1);2*ones(569-mcount,1)];

%% wpdc
data_wpbc = csv2cell('wpbc.csv','fromfile');
mcount = 0;
for i = 1:155
    if data_wpbc{i,33}=='1'
        mcount = mcount+1;
    end
end

data_m = zeros(mcount,32);
data_b = zeros(155-mcount,32);
countm = 1; countb = 1;
for i = 1:155
    if data_wpbc{i,33}=='1'
        for j = 1:32
            data_m(countm,j) = str2double(cell2mat(data_wpbc(i,j)));
        end
        countm = countm+1;
    else
        for j = 1:32
            data_b(countb,j) = str2double(cell2mat(data_wpbc(i,j)));
        end
        countb = countb+1;
    end
end
feature_wpdc = [data_m;data_b];
label_wpdc = [ones(mcount,1);2*ones(155-mcount,1)];

end
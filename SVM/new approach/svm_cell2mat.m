function [feature,label] = svm_cell2mat(noli,setop,corefeat)
%% wbdc
if setop == 1
    data_wdbc = svm_csv2cell('wdbc30.csv','fromfile');
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
    
    label = [ones(mcount,1);2*ones(569-mcount,1)];
    if corefeat == 0
        if noli == 0
            feature = [data_m;data_b];
        else
            tmp = [data_m;data_b];
            [len,~] = size(tmp);
            feature = (tmp-repmat(min(tmp),[len 1]))./repmat(max(tmp),[len 1]);
        end
    else
        if noli == 0
            feature = [data_m(:,1:10);data_b(:,1:10)];
        else 
            tmp = [data_m(:,1:10);data_b(:,1:10)];
            [len,~] = size(tmp);
            feature = (tmp-repmat(min(tmp),[len 1]))./repmat(max(tmp),[len 1]);
        end
    end
end

%% wpdc 30
if setop == 2
    data_wpbc = svm_csv2cell('wpbc30.csv','fromfile');
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
    
    label = [ones(mcount,1);2*ones(155-mcount,1)];
    if corefeat == 0
        if noli == 0
            feature = [data_m;data_b];
        else
            tmp = [data_m;data_b];
            [len,~] = size(tmp);
            feature = (tmp-repmat(min(tmp),[len 1]))./repmat(max(tmp),[len 1]);
        end
    else
        if noli == 0
            feature = [data_m(1:10) data_m(31:32);data_b(1:10) data_b(31:32)];
        else
            tmp = [data_m(1:10) data_m(31:32);data_b(1:10) data_b(31:32)];
            [len,~] = size(tmp);
            feature = (tmp-repmat(min(tmp),[len 1]))./repmat(max(tmp),[len 1]);
        end
    end
end
function svm_model_construct(feature,label,featrd)
% all models
% 1: L1 norm linear kernal
% 2: L1 norm rbf kernal
% 3: L1 norm polynomial kernal
% 4: L1 norm non-unicost linear kernel
% 5: L2 norm linear kernal
% 6: L2 norm rbf kernal

kfold = 3;
worker = 4; % set times of repeat

for model = 1:6
    %% model 1
    if model == 1
        which_C_to_start = -5;
        which_C_to_end = 15;
        c_iter = zeros(1,(which_C_to_end-which_C_to_start+1));
        for i = 1:(which_C_to_end-which_C_to_start+1)
            c_iter(1,i) = 2^(which_C_to_start+i-1);
        end
        c_iter = repmat(c_iter,[worker 1]); 
        CCR_model_1 = zeros(worker,(which_C_to_end-which_C_to_start+1));
        recall_model_1 = zeros(worker,(which_C_to_end-which_C_to_start+1));
        
        parfor i = 1:worker
            [CCR_model_1(i,:),recall_model_1(i,:)] = ...
                svm_train_test(featrd,kfold,'linear','SMO',0,c_iter(i,:),feature,label);
        end
        CCR_ave = mean(CCR_model_1); 
        CCR_maxi = max(CCR_model_1); 
        CCR_mini = min(CCR_model_1);
        recell_ave = mean(recall_model_1); 
        recell_maxi = max(recall_model_1); 
        recell_mini = min(recall_model_1);
        
        fig11 = figure;
        errorbar(which_C_to_start:which_C_to_end,CCR_ave,CCR_ave-CCR_mini,CCR_maxi-CCR_ave);
        xlabel('C = 2^c'); ylabel('CCR'); title('L1 norm linear kernal');
        
        fig12 = figure;
        errorbar(which_C_to_start:which_C_to_end,CCR_ave,recell_ave-recell_mini,recell_maxi-recell_ave);
        xlabel('C = 2^c'); ylabel('recall'); title('L1 norm linear kernal');
        
        save('model_1.mat','CCR_model_1','CCR_ave','CCR_maxi','CCR_mini',...
            'recall_model_1','recell_ave','recell_maxi','recell_mini');
        saveas(fig11,'CCR_model_1.fig'); saveas(fig12,'recall_model_1.fig');
        
        disp(sprintf('L1 norm linear kernal complete'));
    end
    
    %% model 2
    if model == 2
        which_C_to_start = -5;
        which_C_to_end = 15;
        c_iter = zeros(1,(which_C_to_end-which_C_to_start+1));
        for i = 1:(which_C_to_end-which_C_to_start+1)
            c_iter(1,i) = 2^(which_C_to_start+i-1);
        end
        c_iter = repmat(c_iter,[worker 1]);
        
        which_sigma_to_start = -7;
        which_sigma_to_end = 7;
        sigma_iter = zeros(1,(which_sigma_to_end-which_sigma_to_start+1));
        for i = 1:(which_sigma_to_end-which_sigma_to_start+1)
            sigma_iter(1,i) = 2^(which_sigma_to_start+i-1);
        end
        sigma_iter = repmat(sigma_iter,[worker 1]);
        
        CCR_model_2i = zeros(worker,(which_sigma_to_end-which_sigma_to_start+1),...
            (which_C_to_end-which_C_to_start+1));
        
        recall_model_2i = zeros(worker,(which_sigma_to_end-which_sigma_to_start+1),...
            (which_C_to_end-which_C_to_start+1));
        
        parfor i = 1:worker
            [CCR_model_2i(i,:,:),recall_model_2i(i,:,:)] = svm_train_test(featrd,kfold,'rbf','SMO',sigma_iter(i,:),c_iter(i,:),...
                feature,label);
        end
        CCR_ave = squeeze(sum(CCR_model_2i)/worker);
        CCR_maxi = squeeze(max(CCR_model_2i));
        CCR_mini = squeeze(min(CCR_model_2i));
        recall_ave = squeeze(sum(recall_model_2i)/worker);
        recall_maxi = squeeze(max(recall_model_2i));
        recall_mini = squeeze(min(recall_model_2i));
        
        fig21 = figure;
        [X,Y] = meshgrid(which_sigma_to_start:which_sigma_to_end, which_C_to_start:which_C_to_end);
        mesh(X',Y',CCR_ave); hold on;
        mesh(X',Y',CCR_maxi); mesh(X',Y',CCR_mini); hold off
        xlabel('Sigma = 2^s'); ylabel('C = 2^c'); zlabel('CCR'); title('L1 norm rbf kernal');
        fig22 = figure;
        [X,Y] = meshgrid(which_sigma_to_start:which_sigma_to_end, which_C_to_start:which_C_to_end);
        mesh(X',Y',recall_ave); hold on;
        mesh(X',Y',recall_maxi); mesh(X',Y',recall_mini); hold off
        xlabel('Sigma = 2^s'); ylabel('C = 2^c'); zlabel('recall'); title('L1 norm rbf kernal');
        
        save('model_2.mat','CCR_model_2i','CCR_ave','CCR_maxi','CCR_mini'...
            ,'recall_model_2i','recall_ave','recall_maxi','recall_mini');
        saveas(fig21,'CCR_model_2.fig'); saveas(fig22,'recall_model_2.fig');
         
        disp(sprintf('L1 norm rbf kernal complete'));
    end
     
    %% model 5
    if model == 5
        which_C_to_start = -5;
        which_C_to_end = 15;
        c_iter = zeros(1,(which_C_to_end-which_C_to_start+1));
        for i = 1:(which_C_to_end-which_C_to_start+1)
            c_iter(1,i) = 2^(which_C_to_start+i-1);
        end
        c_iter = repmat(c_iter,[worker 1]); 
        CCR_model_5 = zeros(worker,(which_C_to_end-which_C_to_start+1));
        recall_model_5 = zeros(worker,(which_C_to_end-which_C_to_start+1));
        
        parfor i = 1:worker
            [CCR_model_5(i,:),recall_model_5(i,:)] = svm_train_test(featrd,kfold,'linear','QP',0,c_iter(i,:),feature,label);
        end
        CCR_ave = mean(CCR_model_5); 
        CCR_maxi = max(CCR_model_5); 
        CCR_mini = min(CCR_model_5);
        recell_ave = mean(recall_model_5); 
        recell_maxi = max(recall_model_5); 
        recell_mini = min(recall_model_5);
        
        fig51 = figure;
        errorbar(which_C_to_start:which_C_to_end,CCR_ave,CCR_ave-CCR_mini,CCR_maxi-CCR_ave);
        xlabel('C = 2^c'); ylabel('CCR'); title('L1 norm linear kernal');
        
        fig52 = figure;
        errorbar(which_C_to_start:which_C_to_end,CCR_ave,recell_ave-recell_mini,recell_maxi-recell_ave);
        xlabel('C = 2^c'); ylabel('recall'); title('L1 norm linear kernal');
        
        save('model_5.mat','CCR_model_5','CCR_ave','CCR_maxi','CCR_mini',...
            'recall_model_5','recell_ave','recell_maxi','recell_mini');
        saveas(fig51,'CCR_model_5.fig'); saveas(fig52,'recall_model_5.fig');
        
        disp(sprintf('L2 norm linear kernal complete'));
    end
    
    %% model 6
    if model == 6
        which_C_to_start = -5;
        which_C_to_end = 15;
        c_iter = zeros(1,(which_C_to_end-which_C_to_start+1));
        for i = 1:(which_C_to_end-which_C_to_start+1)
            c_iter(1,i) = 2^(which_C_to_start+i-1);
        end
        c_iter = repmat(c_iter,[worker 1]);
        
        which_sigma_to_start = -7;
        which_sigma_to_end = 7;
        sigma_iter = zeros(1,(which_sigma_to_end-which_sigma_to_start+1));
        for i = 1:(which_sigma_to_end-which_sigma_to_start+1)
            sigma_iter(1,i) = 2^(which_sigma_to_start+i-1);
        end
        sigma_iter = repmat(sigma_iter,[worker 1]);
        
        CCR_model_6i = zeros(worker,(which_sigma_to_end-which_sigma_to_start+1),...
            (which_C_to_end-which_C_to_start+1));
        
        recall_model_6i = zeros(worker,(which_sigma_to_end-which_sigma_to_start+1),...
            (which_C_to_end-which_C_to_start+1));
        
        parfor i = 1:worker
            [CCR_model_6i(i,:,:),recall_model_6i(i,:,:)] = svm_train_test(featrd,kfold,'rbf','QP',sigma_iter(i,:),...
                c_iter(i,:),feature,label);
        end
        CCR_ave = squeeze(sum(CCR_model_6i)/worker);
        CCR_maxi = squeeze(max(CCR_model_6i));
        CCR_mini = squeeze(min(CCR_model_6i));
        recall_ave = squeeze(sum(recall_model_6i)/worker);
        recall_maxi = squeeze(max(recall_model_6i));
        recall_mini = squeeze(min(recall_model_6i));
        
        fig61 = figure;
        [X,Y] = meshgrid(which_sigma_to_start:which_sigma_to_end, which_C_to_start:which_C_to_end);
        mesh(X',Y',CCR_ave); hold on;
        mesh(X',Y',CCR_maxi); mesh(X',Y',CCR_mini); hold off
        xlabel('Sigma = 2^s'); ylabel('C = 2^c'); zlabel('CCR'); title('L2 norm rbf kernal');
        fig62 = figure;
        [X,Y] = meshgrid(which_sigma_to_start:which_sigma_to_end, which_C_to_start:which_C_to_end);
        mesh(X',Y',recall_ave); hold on;
        mesh(X',Y',recall_maxi); mesh(X',Y',recall_mini); hold off
        xlabel('Sigma = 2^s'); ylabel('C = 2^c'); zlabel('recall'); title('L2 norm rbf kernal');
        
        save('model_6.mat','CCR_model_6i','CCR_ave','CCR_maxi','CCR_mini'...
            ,'recall_model_6i','recall_ave','recall_maxi','recall_mini');
        saveas(fig61,'CCR_model_6.fig'); saveas(fig62,'recall_model_6.fig');
        
        disp(sprintf('L2 norm rbf kernal complete'));
    end
end
end

%     
%     %% model 3
%     if model == 3
%         %     which_C_to_start = -4;
%         %     which_C_to_end = -2;
%         %     c_iter = zeros(1,(which_C_to_end-which_C_to_start+1));
%         %     for i = 1:(which_C_to_end-which_C_to_start+1)
%         %         c_iter(1,i) = 2^(which_C_to_start+i-1);
%         %     end
%         %     c_iter = repmat(c_iter,[worker 1]);
%         %
%         %     which_pdeg_to_start = 4;
%         %     which_pdeg_to_end = 5;
%         %     pdeg_iter = zeros(1,(which_pdeg_to_end-which_pdeg_to_start+1));
%         %     for i = 1:(which_pdeg_to_end-which_pdeg_to_start+1)
%         %         pdeg_iter(1,i) = 2^(which_pdeg_to_start+i-1);
%         %     end
%         %     pdeg_iter = repmat(pdeg_iter,[worker 1]);
%         %
%         %     CCR_model_3i = zeros(worker,(which_pdeg_to_end-which_pdeg_to_start+1),...
%         %         (which_C_to_end-which_C_to_start+1));
%         %
%         %     parfor i = 1:worker
%         %         [CCR_model_3i(i,:,:)] = svm_train_test(featrd,kfold,'polynomial','SMO',pdeg_iter(i,:),0,c_iter(i,:)...
%         %             ,1,feature,label);
%         %     end
%         %     ave = squeeze(sum(CCR_model_2i)/worker);
%         %     maxi = squeeze(max(CCR_model_2i));
%         %     mini = squeeze(min(CCR_model_2i));
%         %
%         %     fig3 = figure;
%         %     [X,Y] = meshgrid(which_pdeg_to_start:which_pdeg_to_end, which_C_to_start:which_C_to_end);
%         %     contourf(X',Y',ave);
%         %     xlabel('Polynomial degree = 2^s'); ylabel('C = 2^c'); title('L1 norm poly kernal');
%         %
%         %     save('model_3.mat','CCR_model_3i','ave','maxi','mini');
%         %     saveas(fig3,'model_3.fig');
%         
%         disp(sprintf('L1 norm polynomial kernal complete'));
%     end
%     
%     %% model 4
%     if model == 4
%             which_C_to_start = -5;
%             which_C_to_end = 15;
%             c_iter = zeros(1,(which_C_to_end-which_C_to_start+1));
%             for i = 1:(which_C_to_end-which_C_to_start+1)
%                 c_iter(1,i) = 2^(which_C_to_start+i-1);
%             end
%             c_iter = repmat(c_iter,[worker 1]);
%             CCR_model_4 = zeros(worker,(which_C_to_end-which_C_to_start+1));
%         
%             parfor i = 1:worker
%                 [CCR_model_4(i,:)] = svm_train_test(featrd,kfold,'linear','SMO',0,0,c_iter(i,:),1.5,feature,label);
%             end
%             ave = mean(CCR_model_4);
%             maxi = max(CCR_model_4);
%             mini = min(CCR_model_4);
%         
%             fig4 = figure;
%             errorbar(which_C_to_start:which_C_to_end,ave,ave-mini,maxi-ave);
%             xlabel('C = 2^c'); ylabel('CCR'); title('L1 norm non-unicost linear kernel');
%         
%             save('model_4.mat','CCR_model_4','ave','maxi','mini');
%             saveas(fig4,'model_4.fig');
%         
%             disp(sprintf('L1 norm linear kernal with non-uniform cost complete'));
%     end
%
% PCA was performed on the selected GHSI categories and demographic
% predictors
% PCA(detect + respond + health + risk + 18 demo variables)

clear 
load EuropeCountries36TranformedData.mat

rng default

mr = values(:,1); 

data_mat = values (:,[4,5,6,8,10:end]) ; 
data_varnames = variableNames([4,5,6,8,10:end]) ; 
mr_varname = variableNames (1) ; 

data_mat_norm = normalize(data_mat) ; 

%PCs explaining the predefined proportion of cumulative variance (85%) were retained for subsequent analysis.

[coeff,score, ~,~,explained] = pca(data_mat_norm) ; 
disp(sum(explained(1:8)))
demo_pc = score(:,1:8) ;

data_mat = demo_pc ; 
data_varnames = {'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8'} ; 

%% Correlation matricides between the selected PCs and the original variables 
% generated to  identify the variables contributing to each component.
% Save correlation coefficients tables to one Excel file

[R_all,P_all] = corrcoef([demo_pc,data_mat_norm]) ; 
R = R_all(9:end,1:8) ; 
P = P_all(9:end,1:8) ; 

corr_coef_table_v5 = array2table(R,'VariableNames',{'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8'},...
    'RowNames',data_varnames) 

corr_pVal_table_v5 = array2table(P,'VariableNames',{'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8'},...
    'RowNames',data_varnames)

save PC_correlation_tables corr_coef_table_v5 corr_pVal_table_v5 
writetable(corr_coef_table_v5, 'PC_Corrcoefs.xlsx','Sheet',1); 
writetable(corr_pVal_table_v5, 'PC_Corrcoefs.xlsx','Sheet',2);

%% For each retained PC, the top 3 variables with the highest absolute
% loadings were identified as the main contributors to that component.

numPC = size(coeff,2);

coeff_sorted = zeros(size(coeff));

for i = 1:size(coeff,2)
    [~, idx] = sort(abs(coeff(:,i)), 'descend');
    coeff_sorted(:,i) = coeff(idx,i);

end


for i = 1:size(coeff,2)

    abs_loadings = sort(abs(coeff(:,i)), 'descend');

    cumulative = cumsum(abs_loadings) / sum(abs_loadings);

    fprintf('PC%d:\n', i)
    fprintf('Top 2: %.1f%%\n', cumulative(2)*100)
    fprintf('Top 5: %.1f%%\n', cumulative(5)*100)
    fprintf('Top 7: %.1f%%\n', cumulative(7)*100)

end


[~,idx] = sort(abs(coeff(:,1)),'descend');
selected_PC1_variables = data_varnames(idx(1:3));

[~,idx] = sort(abs(coeff(:,2)),'descend');
selected_PC2_variables = data_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,3)),'descend');
selected_PC3_variables = data_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,4)),'descend');
selected_PC4_variables = data_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,5)),'descend');
selected_PC5_variables = data_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,6)),'descend');
selected_PC6_variables = data_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,7)),'descend');
selected_PC7_variables = data_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,8)),'descend');
selected_PC8_variables = data_varnames (idx(1:3));


%% Stepwise and relaxed lasso regression of extracted PC with m/r

swlm = stepwiselm(data_mat,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])

[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(data_mat,mr, data_varnames) ; 


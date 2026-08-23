% PCA was performed on the selected GHSI categories (Detect + respond + health) and risk + demographic variables

clear 
load EuropeCountries36TranformedData.mat

%% SW and RL of Detect + respond + health + PCA (risk + 18 demo var)
rng default

mr = values(:,1); 
mr_varname = variableNames(1) ; 

%PC analysis and the selection of PCs explaining the predefined proportion 
% of cumulative variance (85%) for subsequent analysis.

% Index variables PCA (GHSI categories - Detect, Respond, Health)
index = values(:,[4,5,6]); 
index_varnames = variableNames([4,5,6]); 
index_norm = normalize(index);

[~,score,~,~,explained_ind,~] = pca(index_norm) ; 
disp(cumsum(explained_ind))

% First two PCs explain 92% of variance
index_PC1 = score(:,1) ; 
index_PC2 = score(:,2) ; 

% Demographic data PCA
demo_data = values (:,[8,10:end]) ; 
demo_varnames = variableNames ([8,10:end]) ; 
demo_data_norm = normalize(demo_data) ; 

[coeff,score_demo,~,~,explained_demo] = pca(demo_data_norm) ; 
disp(cumsum(explained_demo))

% First 7 PCs explain approximately 85% of variance.
demo_pc = score_demo(:,1:7) ; 


%% For each retained PC, the top 3 variables with the highest absolute
% loading were identified as the main contributors to that component

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
selected_PC1_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,2)),'descend');
selected_PC2_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,3)),'descend');
selected_PC3_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,4)),'descend');
selected_PC4_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,5)),'descend');
selected_PC5_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,6)),'descend');
selected_PC6_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff(:,7)),'descend');
selected_PC7_variables = demo_varnames (idx(1:3));

%% Stepwise linear and relaxed lasso regression of extracted PCs (Index and Demo) with m/r

% Formation of matrix for the regression analysis: 
data_mat = [index_PC1, index_PC2, demo_pc] ; 
data_varnames = {'Index PC1','Index PC2' 'PC1','PC2','PC3','PC4','PC5','PC6','PC7'}; 

% Regression analysis

swlm = stepwiselm(data_mat,mr,'Upper','linear','VarNames',[data_varnames,mr_varname]);

[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(data_mat,mr, data_varnames) ; 


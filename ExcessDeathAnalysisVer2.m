%%  %% Excess and unexplained deaths analysis
% This script performs the complete analysis of excess and unexplained
% COVID-19-related mortality across 35 European countries.

load EuropeExcessDeathsAnalysis.mat

%% Extraction of COVID-19 related parameters 

total_covid_cases = values(:,29); 
total_covid_deaths = values(:,30); 

CFR = total_covid_deaths./total_covid_cases;  
R0 = values (:,28);

max_covid_cases = values(:,31); 
max_covid_deaths = values(:,32);

excess_deaths = values(:,34);
total_deaths = values (:,33); 

population_size = values(:,37);
days=values(:,38);

unexplained_deaths = values(:,35);
relative_excess_deaths =values (:,36);

%% Calculate relative values of additional variables:  
average_daily_covid_cases = total_covid_cases./(population_size.*days) ; % Total cases - Average daily cases (Total cases po danu po stanovniku)
average_daily_covid_deaths  = total_covid_deaths ./(population_size.*days) ; % Total deaths -  Average daily deaths (Total deaths po danu po stanovniku)

max_covid_cases_per_capita = max_covid_cases./population_size ; %Ma0x daily cases po stanovniku

max_deaths_sort = sort(max_covid_deaths) ; % sort data
zero_idx = find(max_covid_deaths == 0);   % Position of zero
max_covid_deaths(zero_idx) = max_deaths_sort(2);   % Replace with second smallest value

max_covid_deaths_per_capita = max_covid_deaths./population_size ; % max daily deaths - Max daily cases po stanovniku

%% Additional data transformation and save as a separate file:
relative_excess_deaths_transf= sqrt(relative_excess_deaths-min(relative_excess_deaths)) ; % relative excess deaths
unexplained_deaths_transf =  sign(unexplained_deaths).*abs(unexplained_deaths).^(1/3); ; % unexplained deaths

CFR_transf= log(CFR) ; % CFR

average_daily_covid_cases_transf  = log(average_daily_covid_cases) ; % Average daily cases (Total cases po danu po stanovniku)
average_daily_covid_deaths_transf = log(average_daily_covid_deaths) ; % Average daily deaths (Total deaths po danu po stanovniku)

max_covid_cases_transf  = log(max_covid_cases_per_capita) ; % Max daily cases po stanovniku
max_covid_deaths_transf  = log(max_covid_deaths_per_capita) ; % Max daily deaths po stanovniku

values_transf = [ ...
    relative_excess_deaths_transf, ...
    unexplained_deaths_transf, ...
    CFR_transf, ...
    average_daily_covid_cases_transf, ...
    average_daily_covid_deaths_transf, ...
    max_covid_cases_transf, ...
    max_covid_deaths_transf];

var_names_transf = { ...
    'relative_excess_deaths_transf',...
    'unexplained_deaths_transf',...
    'CFR_transf',...
    'average_daily_covid_cases_transf',...
    'average_daily_covid_deaths_transf',...
    'max_covid_cases_transf',...
    'max_covid_deaths_transf'};

save('EuropeExcessDeathsAnalysis.mat','values_transf','var_names_transf','-append');

% Univariate analysis
%% PCA of relative/rate-based COVID-19 cases and mortality measures
covid_counts =values_transf (:,[4 5 6 7]) ; 
covid_counts_norm = normalize(covid_counts) ; 

counts_varnames =var_names_transf([4 5 6 7]); 
[coeff_covid_counts, score_covid_counts, ~,~,explained_covid_counts] = pca(covid_counts_norm) ; 
disp(cumsum(explained_covid_counts))

counts_PC1 = score_covid_counts(:,1) ; % 1 PC explained ~ 89% variability 


[pc_corr_covid_counts, pc_p_covid_counts] = corr(score_covid_counts(:,1), covid_counts_norm);

%% PCA of selected GHSI variables

GHSI = values (:,[4:6]) ;  
GHSI_norm = normalize(GHSI) ; 
GHSI_varnames = variableNames ([4:6]) ; 
[coeff_GHSI, score_GHSI, ~,~,explained_GHSI] = pca(GHSI_norm) ; 
disp(cumsum(explained_GHSI));

GHSI_PC1 = score_GHSI(:,1) ; % 74% of variance
GHSI_PC2 = score_GHSI(:,2) ; % 92% of variance

[pc_corr_GHSI, pc_p_GHSI] = corr(score_GHSI(:,1:2),GHSI_norm);

%% PCA of demographic variables to reduce dimensionality and 
% account for correlations among predictors
demo = values(:,[10:27]) ; 
demo_norm = normalize(demo) ; 
demo_varnames = variableNames([10:27]) ; 
[coeff_demo, score_demo, ~,~,explained_demo] = pca(demo_norm) ; 
disp(cumsum(explained_demo)); 

demo_PCs = score_demo(:,1:8) ; % first 8 PCs explain ~ 87% of variance 

[pc_corr_demo, pc_p_demo] = corr(score_demo(:,1:8),demo_norm);

%% Evaluation of which original variables contribute most strongly 
% to each selected demographic PC

numPC = size(coeff_demo,2);

coeff_sorted = zeros(size(coeff_demo));

for i = 1:size(coeff_demo,2)
    [~, idx] = sort(abs(coeff_demo(:,i)), 'descend');
    coeff_sorted(:,i) = coeff_demo(idx,i);

end


for i = 1:size(coeff_demo,2)

    abs_loadings = sort(abs(coeff_demo(:,i)), 'descend');

    cumulative = cumsum(abs_loadings) / sum(abs_loadings);

    fprintf('PC%d:\n', i)
    fprintf('Top 2: %.1f%%\n', cumulative(2)*100)
    fprintf('Top 5: %.1f%%\n', cumulative(5)*100)
    fprintf('Top 7: %.1f%%\n', cumulative(7)*100)

end
% Selected variables for each principal component

[~,idx] = sort(abs(coeff_demo(:,1)),'descend');
selected_PC1_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff_demo(:,2)),'descend');
selected_PC2_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff_demo(:,3)),'descend');
selected_PC3_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff_demo(:,4)),'descend');
selected_PC4_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff_demo(:,5)),'descend');
selected_PC5_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff_demo(:,6)),'descend');
selected_PC6_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff_demo(:,7)),'descend');
selected_PC7_variables = demo_varnames (idx(1:3));

[~,idx] = sort(abs(coeff_demo(:,8)),'descend');
selected_PC8_variables = demo_varnames (idx(1:3));

%% predictor matrix formation

predictor_mat = [CFR_transf R0 counts_PC1 GHSI_PC1 GHSI_PC2 demo_PCs]  ; 
predictor_varnames = {'CFR', 'R0', 'counts PC1', 'GHSI PC1', 'GHSI PC2',...
    'demo PC1','demo PC2','demo PC3','demo PC4','demo PC5','demo PC6',...
    'demo PC7','demo PC8'} ; 
response_varnames = {'excess deaths', 'relative excess deaths', 'unexplained deaths'} ; 

save excess_deaths_PC_predictors predictor_mat predictor_varnames response_varnames excess_deaths unexplained_deaths


%% Assessment of associations between the resulting predictors and relative excess deaths and unexplained deaths

% Relative Excess deaths regressions
% linear regression model
lin_mdl = fitlm(normalize(predictor_mat),relative_excess_deaths_transf,'VarNames',[predictor_varnames,response_varnames(2)]) ; 

%stepwise linear regression
swlm = stepwiselm(normalize(predictor_mat),relative_excess_deaths_transf,'Upper','linear','VarNames',[predictor_varnames,response_varnames(2)]) ; 

%Lasso regression
[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(predictor_mat,relative_excess_deaths_transf, predictor_varnames) ; 

%% Unexplained deaths regressions

% Linear regression model
lin_mdl = fitlm(normalize(predictor_mat),unexplained_deaths_transf,'VarNames',[predictor_varnames,response_varnames(3)]) ;

% Stepwise linear regression
swlm = stepwiselm(normalize(predictor_mat),unexplained_deaths_transf,'Upper','linear','VarNames',[predictor_varnames,response_varnames(3)]) ;

% Lasso regression
[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(predictor_mat,unexplained_deaths_transf, predictor_varnames)  ; 
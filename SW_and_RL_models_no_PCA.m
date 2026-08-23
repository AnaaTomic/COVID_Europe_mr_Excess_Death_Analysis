%% Regression models without PCA
% Stepwise linear regression and LASSO regression were performed on the original transformed GHSI and demographic variables.
% Predictors and model performance were evaluated and recorded for comparison between the two approaches.

%% 1) detect + respond + health + risk + 18 demo variables 

clear
load EuropeCountries36TranformedData.mat 
rng default

mr = values (:,1); 
mr_varname = variableNames(1) ; 
data_mat = values(:,[4,5,6,8,10:end]); 
data_varnames = variableNames([4,5,6,8,10:end]); 

swlm = stepwiselm(data_mat,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])

[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(data_mat,mr, data_varnames) 
%% 2) Overall GHSI + 18 demo variables: 

clear
load EuropeCountries36TranformedData.mat 
rng default

mr = values (:,1); 
mr_varname = variableNames(1) ; 
data_mat = values(:,[2,10:end]) ; 
data_varnames = variableNames ([2,10:end]) ; 

swlm = stepwiselm(data_mat,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])
[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(data_mat,mr, data_varnames)  

%% 3) Covid + 18 demo variables

clear
load EuropeCountries36TranformedData.mat 
rng default

mr = values (:,1); 
mr_varname = variableNames(1) ; 
data_mat = values (:,[9:end]) ; 
data_varnames = variableNames ([9:end]) ;

swlm = stepwiselm(data_mat,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])
[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ] = relaxedlasso (data_mat,mr, data_varnames)

%% 4) prevent + norms + demo variables

clear
load EuropeCountries36TranformedData.mat 
rng default

mr = values (:,1); 
mr_varname = variableNames(1) ; 
data_mat = values (:,[3,7,10:end]) ; 
data_varnames = variableNames ([3,7,10:end]) ;

swlm = stepwiselm(data_mat,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])
[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ] = relaxedlasso (data_mat,mr, data_varnames)
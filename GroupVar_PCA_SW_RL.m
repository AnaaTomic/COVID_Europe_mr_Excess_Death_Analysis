
%% PCA and regression analysis of conceptually grouped variables

% PCA: Variables were grouped according to predefined conceptual domains,
% including country prosperity, demographic/age structure, healthcare
% preparedness, and cardio-metabolic risk. PCs explaining at least 80% of the cumulative
% variance within each group were retained for subsequent analysis

clear
load EuropeCountries36TranformedData.mat
rng default

mr = values(:,1) ; 
mr_varname = variableNames(1) ; 


%Prosperity 
prosperity = values(:,[8,13,14,15]) ; 
prosperity_Varnames = {'Risk','IM','GDP','HDI'} ; 
prosperity_norm = normalize(prosperity) ; 
[~,score_prosperity,~,~,explained_prosperity,~] = pca(prosperity_norm) ; 
disp(cumsum(explained_prosperity));
% First PC explained ~ 87% of variance 
prosperity_PC1 = score_prosperity(:,1) ; 

%Cardiometabolic Risk Factors 
Cardiometabolic = values(:,[18,22,23]) ; 
Cardiometabolic_Varnames = {'CH','CD','RBP'}  ; 
Cardiometabolic_norm = normalize(Cardiometabolic) ; 
[~,score_Cardiometabolic,~,~,explained_Cardiometabolic,~] = pca(Cardiometabolic_norm) ; 
disp(cumsum(explained_Cardiometabolic));
% Two PCs explained ~ 93% of variance
Cardiometabolic_PC1 = score_Cardiometabolic(:,1) ; 
Cardiometabolic_PC2 = score_Cardiometabolic(:,2) ; 

%Age
age = values(:,[10,12,19]) ; 
age_Varnames = {'UP','MA','AL'} ; 
age_norm = normalize(age) ; 
[~,score_age,~,~,explained_age,~] = pca(age_norm) ; 
disp(cumsum(explained_age))
% Two PCs explained ~ 83% of variance
age_PC1 = score_age(:,1) ; 
age_PC2 = score_age(:,2) ; 

% Healthcare system preparedness 

Healthcare = values(:,[4,5,6]) ; 
Healthcare_Varnames = {'Detect','Respond','Health'}  ; 
Healthcare_norm = normalize(Healthcare) ; 
[~,score_Healthcare,~,~,explained_Healthcare,~] = pca(Healthcare_norm) ; 
disp(cumsum(explained_Healthcare));
% Two PCs explained ~ 92% of variance
Healthcare_PC1 = score_Healthcare(:,1) ; 
Healthcare_PC2 = score_Healthcare(:,2) ; 

%% Health-related habits (not used)

habits = values(:,[19,21,20,24]) ; 
habits_Varnames = {'AL','OB','SM','IN'}  ; 
habits_norm = normalize(habits) ; 
[~,score_habits,~,~,explained_habits,~] = pca(habits_norm) ; 
disp(cumsum(explained_habits));


%% Preventive Public Health Capacity (not used)

PublicHealth = values(:,[3,7,25]) ; 
PublicHealth_Varnames = {'Prevent','Norms','BSG'}  ; 
PublicHealth_norm = normalize(PublicHealth) ; 
[~,score_PublicHealth,~,~,explained_PublicHealth,~] = pca(PublicHealth_norm) ; 
disp(cumsum(explained_PublicHealth));

PublicHealth_PC1 = score_PublicHealth(:,1) ; 
PublicHealth_PC2 = score_PublicHealth(:,2) ; 


%% Variables that were not assigned to a conceptual group were retained
% and entered separately as individual predictors

other_variables = values(:,[11,16,17,20,21,24,25,26,27]) ; 
other_varnames = variableNames([11,16,17,20,21,24,25,26,27]) ; 

data_mat = [prosperity_PC1,Cardiometabolic_PC1,Cardiometabolic_PC2, age_PC1,age_PC2,Healthcare_PC1, Healthcare_PC2, other_variables] ; 
data_mat_norm = normalize(data_mat) ; 
data_varnames = [{'prosperity PC1', 'Cardiometabolic PC1','Cardiometabolic PC2','age PC1','age PC2','Healthcare PC1','Healthcare PC2'},other_varnames] ;  

%% The resulting group-specific PCs and individual predictors were used
% in stepwise linear regression and Relaxed LASSO models with m/r as a response variable.

% Stepwise linear regression: 
swlm = stepwiselm(data_mat_norm,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])

% Lasso regression: 
[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(data_mat,mr, data_varnames) ; 

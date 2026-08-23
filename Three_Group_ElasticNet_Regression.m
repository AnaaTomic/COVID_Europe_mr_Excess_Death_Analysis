%% PCA and Elastic Net regression analysis of conceptually grouped variables

clear
load EuropeCountries36TranformedData.mat
rng default

%Prosperity 
prosperity = values(:,[8,13,14,15]) ; 
prosperity_Varnames = {'Risk','IM','GDP','HDI'} ; 
prosperity_norm = normalize(prosperity) ; 
[~,score_prosperity,~,~,explained_prosperity,~] = pca(prosperity_norm) ; 
disp(cumsum(explained_prosperity));
prosperity_PC1 = score_prosperity(:,1) ; 

%Cardiometabolic Risk Factors 
Cardiometabolic = values(:,[18,22,23]) ; 
Cardiometabolic_Varnames = {'CH','CD','RBP'}  ; 
Cardiometabolic_norm = normalize(Cardiometabolic) ; 
[~,score_Cardiometabolic,~,~,explained_Cardiometabolic,~] = pca(Cardiometabolic_norm) ; 
disp(cumsum(explained_Cardiometabolic));

Cardiometabolic_PC1 = score_Cardiometabolic(:,1) ; 
Cardiometabolic_PC2 = score_Cardiometabolic(:,2) ; 

%Age
age = values(:,[10,12,19]) ; 
age_Varnames = {'UP','MA','AL'} ; 
age_norm = normalize(age) ; 
[~,score_age,~,~,explained_age,~] = pca(age_norm) ; 
disp(cumsum(explained_age))

age_PC1 = score_age(:,1) ; 
age_PC2 = score_age(:,2) ; 

% Healthcare system preparedness 

Healthcare = values(:,[4,5,6]) ; 
Healthcare_Varnames = {'Detect','Respond','Health'}  ; 
Healthcare_norm = normalize(Healthcare) ; 
[~,score_Healthcare,~,~,explained_Healthcare,~] = pca(Healthcare_norm) ; 
disp(cumsum(explained_Healthcare));

Healthcare_PC1 = score_Healthcare(:,1) ; 
Healthcare_PC2 = score_Healthcare(:,2) ; 

other_variables = values(:,[11,16,17,20,21,24,25,26]) ; 
other_varnames = variableNames([11,16,17,20,21,24,25,26]) ; 


mr = values(:,1) ; 
mr_varname = variableNames(1) ; 

data_mat = [prosperity_PC1,Cardiometabolic_PC1,Cardiometabolic_PC2, age_PC1,age_PC2,Healthcare_PC1, Healthcare_PC2, other_variables] ; 
data_mat_norm = normalize(data_mat) ; 
data_varnames = [{'prosperity PC1', 'Cardiometabolic PC1','Cardiometabolic PC2','age PC1','age PC2','Healthcare PC1','Healthcare PC2'},other_varnames] ;  

matrix_all = data_mat ; 
Varnames = data_varnames ; 
mean_matrix_all = mean(matrix_all);
std_matrix_all = std(matrix_all);
matrix_all_norm = (matrix_all - mean_matrix_all)./std_matrix_all;

alpha_vec = 0.01:0.01:1         ;

% Scan for lambda values: 
i = 0 ;
for alpha = alpha_vec
    i = i + 1   ;
    [B_scan,FitInfo_scan] = lasso(matrix_all_norm,mr,'Alpha',alpha);
    lambda_vec{i} = FitInfo_scan.Lambda   ;
end

% Cross-validation parameters:
n_fold = 5;
n_repart = 40;
cv = cvpartition(mr,'KFold',n_fold,'Stratify',false);

MSE_alpha_lambda_mat = [];

% Create hyperparameter combination matrix: 
param_matrix = [];

for i = 1:length(alpha_vec)

    alpha = alpha_vec(i);

    for lambda = lambda_vec{i}

        param_matrix = [param_matrix [alpha; lambda]];

    end

end

lena_param = size(param_matrix,2);
param_index_perm_vec = randperm(lena_param);

% Cross-validation: 

for index = param_index_perm_vec 

    alpha = param_matrix(1,index);
    lambda = param_matrix(2,index);

    SE_vector_train = [];
    SE_vector_test = [];

    for repart = 1:n_repart 

        cnew = repartition(cv);

        parfor  i = 1:n_fold

            index_train = find(cnew.test(i)==0);
            index_test = find(cnew.test(i));
            
            mr_train = mr(index_train);
            matrix_all_train = matrix_all(index_train,:);
            mean_matrix_all_train = mean( matrix_all_train );
            std_matrix_all_train = std( matrix_all_train  );
            matrix_all_train_norm = (matrix_all_train - mean_matrix_all_train )./std_matrix_all_train;
            
            mr_test = mr(index_test);
            matrix_all_test = matrix_all(index_test,:);
            matrix_all_test_norm = (  matrix_all_test -  mean_matrix_all_train )./std_matrix_all_train;
            
            [B_train,FitInfo_train] = lasso(matrix_all_train_norm,mr_train,'Alpha',alpha,'Lambda',lambda,'PredictorNames',Varnames);
            coef_train = B_train;
            coef0_train = FitInfo_train.Intercept;
            mrhat_train = matrix_all_train_norm*coef_train + coef0_train;
            MSE_train_norm = sum((mr_train - mrhat_train).^2)/sum((mr_train - mean(mr_train)).^2);
            mrhat_test = matrix_all_test_norm*coef_train + coef0_train  ;
            MSE_test_norm = sum((mr_test - mrhat_test).^2)/sum((mr_test - mean(mr_test)).^2);

            SE_vector_train = [SE_vector_train MSE_train_norm];
            SE_vector_test = [SE_vector_test MSE_test_norm];

        end

    end

    MSE_train = mean(SE_vector_train);
    MSE_test = mean(SE_vector_test);
    sterr_MSE_train = 2*std(SE_vector_train)/sqrt(length(SE_vector_train));
    sterr_MSE_test = 2*std(SE_vector_test)/sqrt(length(SE_vector_test));
    MSE_alpha_lambda_mat = [MSE_alpha_lambda_mat [alpha;lambda;MSE_train;sterr_MSE_train;MSE_test;sterr_MSE_test]];

end

% Finding best models: 

% Finding the min MSE model: 
[MSE_test_min,index_min] = min(MSE_alpha_lambda_mat(5,:));

% Extractiong the min MSE standard error: 
sterr_MSE_test_min =  MSE_alpha_lambda_mat(6,index_min);

% Finding models with MSE within 1 sterr from min MSE: 
MSE_test_min_err = MSE_test_min + sterr_MSE_test_min;
index_min_all = find(MSE_alpha_lambda_mat(5,:) < MSE_test_min_err);
MSE_min_all_test = MSE_alpha_lambda_mat(5,index_min_all); 
sterr_MSE_min_all_test = MSE_alpha_lambda_mat(6,index_min_all); 
MSE_min_all_train = MSE_alpha_lambda_mat(3,index_min_all ); 
sterr_MSE_min_all_train = MSE_alpha_lambda_mat(4,index_min_all); 
lambda_min_all =  MSE_alpha_lambda_mat(2,index_min_all);
alpha_min_all = MSE_alpha_lambda_mat(1,index_min_all); 

matrix_all_norm = normalize(matrix_all) ; 
sp = zeros(1,length(lambda_min_all)) ; 

for i = 1:length(lambda_min_all) 
    % Elastic Net regresison with selected alpha-lambda combination    
    [B,~] = lasso(matrix_all_norm,mr,'Alpha',alpha_min_all(i),'Lambda',lambda_min_all(i),'PredictorNames',Varnames);
    % calculationg the sum of predictors with coeff = 0
    sp(i) = sum(B == 0) ;           
end

% Finding the sparsest model:

[max_sp, index_max_sp] = max(sp); 
lambda_max_sparse = lambda_min_all(index_max_sp); 
alpha_max_sparse = alpha_min_all(index_max_sp); 
MSE_max_sparse_test = MSE_min_all_test(index_max_sp); 
sterr_MSE_sparse_test = sterr_MSE_min_all_test(index_max_sp); 
MSE_max_sparse_train = MSE_min_all_train(index_max_sp); 
sterr_MSE_sparse_train = sterr_MSE_min_all_train(index_max_sp); 

[B,FitInfo] = lasso(matrix_all_norm,mr,'Alpha',alpha_max_sparse,'Lambda',lambda_max_sparse,'PredictorNames',Varnames);
coef = B;           
coef0 = FitInfo.Intercept;
mrhat = matrix_all_norm*coef + coef0;         
R_square = 1 - (sum((mr - mrhat).^2)/sum((mr - mean(mr)).^2 )); 

% Display the results: 

ModelPredictors = FitInfo.PredictorNames(B(:)~=0); 
koeficijenti = B(B(:)~=0);
Intercept = FitInfo.Intercept;

% Bar graph:
figure
cat = categorical(ModelPredictors); 
cat = reordercats(cat, ModelPredictors);
BFC = [125/255 131/255 128/255]; 
BEC = [23/255 28/255 233/255]; 
figure
bar(cat, koeficijenti,'FaceColor', BFC,'EdgeColor',BEC,'LineWidth',1.5);
ylabel('Model coefficients')
title('Elastic net regression (sparse)')
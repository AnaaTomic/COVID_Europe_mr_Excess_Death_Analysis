# COVID_Europe_mr_Excess_Death_Analysis

A series of analysis of COVID-19 severity and excess deaths prediction in 36 Europe countries with GHSI variables and 18 demographic variables was performed.
COVID severity (m/r) analysis 
1.	Dataset: EuropeCountries36TranformedData.mat
Contains the transformed dataset used for the COVID-19 severity (m/r) analysis. The dataset includes:
•	COVID-19 m/r, used as the dependent variable
•	Transformed GHSI category variables
•	Transformed demographic variables

2.	Pearson Correlation Analysis: mr_Pearson_Correlation.m 
This script performs the initial correlation analysis between the m/r and the available predictor variables. The script include Pearson correlation analysis of all available variables, as well as assessment of correlations between m/r and selected GHSI categories. 
The script produces correlation results used for preliminary assessment of associations between the predictors and COVID-19 severity by generating tables containing the Pearson correlation coefficients (R) and corresponding p-values (P), visualization of the correlation matrix using a heatmap, as well as assessment of multicollinearity among predictor variables using variance inflation factor (VIF) and tolerance values.
3.	Regression Models Without PC: SW_and_RL_models_no_PCA.m 
This script evaluates regression models using the original predictor variables without prior dimensionality reduction by PCA. Stepwise linear regression and LASSO regression are performed, producing the selected predictors and model performance measures of the models.
4.	PCA-Based Regression Models: SW_RL_PCA.m; Index_Demo_PCA_SW_RL.m 
PCA was performed on the selected GHSI categories and demographic variables to reduce the number of correlated predictors and identify the main underlying dimensions of the data. Principal components were selected based on the proportion of variance explained (cut-off at 80%). 
The retained components were correlated with the original variables to identify the variables contributing to each component. Moreover, for each retained PC, the top 3 variables with the highest absolute loadings were identified as the main contributors to that component.
Stepwise linear regression and LASSO regression was performed using the resulting principal components.
5.	Grouped PCA Regression:  GroupVar_PCA_SW_RL, GroupVar_ElasticNet_Regression.m 
Variables were organized into conceptually related domains, such as country prosperity, age/demographic structure, healthcare preparedness, and cardiometabolic risk. PCA was performed separately within each predefined domain to reduce correlated variables while preserving the conceptual structure of the predictor groups. Principal components explaining at least 80% of the cumulative variance within each group were retained for subsequent analysis. Variables that did not belong to a conceptual group were retained as individual predictors rather than included in PCA.
The resulting group-specific PCs and individual predictors were used as predictors in stepwise linear regression, Relaxed LASSO and Elastic Net regression to assess their association with m/r as dependent variable. Model performance and selected predictors were evaluated for both approaches and used for the final model selection. 

Excess Deaths analysis 
1.	Dataset: EuropeExcessDeathsAnalysisOn36Countries.mat 
The dataset includes:
•	Transformed GHSI category variables
•	Transformed demographic variables
•	Untransformed COVID-19 case parameters
•	Untransformed COVID-19 death parameters
This file is used as the primary input for the excess deaths analysis.
2.	Excess Deaths Analysis: ExcessDeathAnalysisVer2.m 
This script performs the complete analysis of excess and unexplained COVID-19-related mortality across the 35 European countries.
The analysis includes:
Extraction of relevant COVID-19 case and death parameters from the dataset
Calculation of relative/rate-based COVID-19 and mortality measures, including relative excess deaths and unexplained deaths
Data transformation of variables where required to improve their distribution and suitability for subsequent analyses
PCA of selected COVID-19 measures, GHSI and demographic variables to reduce dimensionality and account for correlations among predictors
Assessment of associations between the resulting predictors and relative excess deaths and unexplained deaths
Stepwise linear regression and Relaxed LASSO regression to identify predictors associated with the relevant mortality outcomes, including relative excess deaths and unexplained deaths.

Used MATLAB functions:
1.	relaxedlasso.m 
This function implements the Relaxed LASSO regression used in the analyses.


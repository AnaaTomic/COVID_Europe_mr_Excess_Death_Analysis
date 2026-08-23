% Initial correlation analysis between the m/r and the available predictor variables

load EuropeCountries36TranformedData.mat

% Create correlation matrix of all variables and save R and P as tables: 

[R,P] = corrcoef(values);
mr_variables_corrcoef_Table = array2table(R,'VariableNames', variableNames, 'RowNames', variableNames); 
mr_variables_pVal_Table = array2table(P,'VariableNames',variableNames, 'RowNames', variableNames);

save mr_correlation_tables mr_variables_corrcoef_Table mr_variables_pVal_Table
writetable(mr_variables_corrcoef_Table, 'Corrcoefs.xlsx','Sheet',1); 
writetable(mr_variables_pVal_Table, 'Corrcoefs.xlsx','Sheet',2);

% Correlation heatmap

figure;
heatmap(variableNames,variableNames,R);
title('Pearson Correlation Matrix (R)');
figure;
heatmap(variableNames,variableNames,P);
title('Pearson Correlation Matrix (P)');


% Keep only upper triangle
R_upper = triu(R,1);
figure('Position',[100 100 1200 1200])

imagesc(R_upper)

clim([-1 1])
axis square
colormap(flipud(jet))
colorbar

hold on

% Overlay lower triangle mask
mask = tril(true(size(R)));
whiteMask = ones(size(R,1), size(R,2), 3);
h = image(whiteMask);
h.AlphaData = mask;

n = length(variableNames);


for i = 1:n
    for j = 1:n

        if i < j

            if P(i,j) < 0.001
                marker = '***';
            elseif P(i,j) < 0.01
                marker = '**';
            elseif P(i,j) < 0.05
                marker = '*';
            else
                marker = '';
            end

            if ~isempty(marker)
                text(j,i,marker,...
                    'HorizontalAlignment','center',...
                    'VerticalAlignment','middle',...
                    'FontSize',12,...
                    'Color','k')
            end

        end

    end
end


% Labels

n = length(variableNames);

xticks(1:n)
yticks(1:n)

xticklabels(variableNames)
yticklabels(variableNames)

xtickangle(45)
ax = gca;
ax.XAxisLocation = 'top';


title('Pearson Correlation Matrix')

hold off

%% VIF evaluation

X=values(:,[2:end]);
X_var_names = variableNames(2:end)

n = size(X,2);

VIF = zeros(n,1);

for i = 1:n

    y = X(:,i);

    X_other = X;
    X_other(:,i) = [];

    mdl = fitlm(X_other,y);

    R2 = mdl.Rsquared.Ordinary;

    VIF(i) = 1/(1-R2);

end

VIF_table = table(X_var_names',VIF,...
    'VariableNames',{'Variable','VIF'});

disp(VIF_table)

%% Tolerance evaluation

Tolerance = 1./VIF;
Tolerance_table = table(X_var_names',Tolerance,...
    'VariableNames',{'Variable','Tolerance'});

disp(Tolerance_table)

%% Variables to correlate with m/r
predictorNames = {'Detect','Respond','Health','ON','CH','AL'};

% Matrix containing the six predictors

X = values (:,[4 5 6 18 19 26]);
y = values (:,1);

figure('Units','inches','Position',[1 1 7.5 5]);

for i = 1:length(predictorNames)

    x = X(:,i);


    % Pearson correlation
    [R,P] = corr(x,y,'Type','Pearson');

    % Scatter plot
    subplot(2,3,i);
    scatter(x,y,40,'b','filled','MarkerFaceAlpha',0.5);
    hold on;

    % Linear regression line
    p = polyfit(x,y,1);
    x_fit = linspace(min(x),max(x),100);
    y_fit = polyval(p,x_fit);
    plot(x_fit,y_fit,'LineWidth',1.5);

    % Labels
    xlabel(predictorNames{i});
    ylabel('m/r');
    title(predictorNames{i});

    % R and P in lower-right corner
    xlim_values = xlim;
    ylim_values = ylim;

    text(xlim_values(1) + 0.95*diff(xlim_values), ...
        ylim_values(1) + 0.05*diff(ylim_values), ...
        sprintf('R = %.3f\nP = %.3f',R,P), ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment','bottom');

    box on;
    hold off;

end
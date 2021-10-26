import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_highly_correlated_cols( dataframe, threshol_vif=5):
    scaler = StandardScaler()
    x = dataframe.drop(columns = ['ttf'])
    X_scaled = scaler.fit_transform(x)


    # we create a new data frame which will include all the VIFs
    # note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
    # we do not include categorical values for mulitcollinearity as they do not provide much information as numerical ones do
    vif = pd.DataFrame()

    # here we make use of the variance_inflation_factor, which will basically output the respective VIFs
    vif["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    # Finally, I like to include names so it is easier to explore the result
    vif["Features"] = x.columns
    highly_corelated_features = list(vif[vif['VIF'] > threshol_vif]['Features'])
    return highly_corelated_features


def drop_highly_corelated_cols(dataframe, col_list):
    return dataframe.drop(col_list, axis=1)

def dropUnnecessaryColumns(data,columnNameList):
    data = data.drop(columnNameList,axis=1)
    return data
from data_ingestion import load_data,clean_data,fetch_column,split_data
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,RobustScaler,StandardScaler
from sklearn.compose import ColumnTransformer


data_df=load_data()
data_clean_df=clean_data(data_df)
x_train,x_test,y_train,y_test=split_data(data_clean_df)
ordinal_col,nominal_col,robust_col,standard_col=fetch_column(x_train)

def Transform_data(ordinal_col,nominal_col,robust_col,standard_col):
    
     Ordinal_pipeline=Pipeline([
        ('imputetion',SimpleImputer(strategy="most_frequent")),
        ('ordinaltransform',OrdinalEncoder())
        ]) 
     
     onehot_pipeline=Pipeline([
         ('imputetion',SimpleImputer(strategy="most_frequent")),
         ('onehottransform',OneHotEncoder(handle_unknown="infrequent_if_exist",min_frequency=2))
    ])
     
     Transformer=ColumnTransformer(
        transformers=(
        ["ordinal",Ordinal_pipeline,ordinal_col],
        ["onehot",onehot_pipeline,nominal_col],
        ['Sscale',StandardScaler(),standard_col],
        ['Rscale',RobustScaler(),robust_col]
        ),
        remainder="passthrough"
        )
     return Transformer
    
if __name__ =="__main__":
    data_df=load_data()
    data_clean_df=clean_data(data_df)
    x_train,x_test,y_train,y_test=split_data(data_clean_df)
    ordinal_col,nominal_col,robust_col,standard_col=fetch_column(x_train)
    transformer=Transform_data(ordinal_col,nominal_col,robust_col,standard_col)
    x_train_tran = transformer.fit_transform(x_train)
from data_ingestion import load_data,clean_data,split_data
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer


data_df=load_data()
data_clean_df=clean_data(data_df)
x_train,x_test,y_train,y_test=split_data(data_clean_df)

def Transform_data(data_clean_df):
    
    cat_col=data_clean_df.select_dtypes(include="object")
    ordinal_col=['Contract']
    nominal_col=[col for col in cat_col if col not in ordinal_col]
    
    Ordinal_transform=Pipeline([
        ('imputetion',SimpleImputer(strategy="most_frequent")),
        ('ordinaltransform',OrdinalEncoder(),)
        ]) 
    
    onehot_transformer=Pipeline([
         ('imputetion',SimpleImputer(strategy="most_frequent")),
         ('onehottransform',OneHotEncoder(handle_unknown="infrequent_if_exist",min_frequency=2))
    ])
    
    Transformer=ColumnTransformer(
        transformers=(
        ["ordinal",Ordinal_transform,ordinal_col],
        ["onehot",onehot_transformer,nominal_col]
        ),
        remainder="passthrough"
        )
    
    return Transformer
    
if __name__ =="main":
    data_df=load_data()
    data_clean_df=clean_data(data_df)
    x_train,x_test,y_train,y_test=split_data(data_clean_df)
    Transform_data(x_train,x_test,y_train,y_test)
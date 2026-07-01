from data_ingestion import load_data,clean_data,split_data,fetch_column
from data_transform import Transform_data
from model import Model
from evaluate import evaluate_model
from sklearn.pipeline import Pipeline

def main():
    
    data_df = load_data()
    data_clean_df=clean_data(data_df)
    x_train,x_test,y_train,y_test=split_data(data_clean_df)
    ordinal_col,nominal_col,robust_col,standard_col=fetch_column(x_train)
    transformer=Transform_data(ordinal_col,nominal_col,robust_col,standard_col)
    model=Model()
    
    pipeline=Pipeline([
        ('transformer',transformer),
        ('model',model)
    ])
    pipeline.fit(x_train,y_train)
    test_pred=pipeline.predict(x_test) 
    train_pred=pipeline.predict(x_train)
    
    
    metrics=evaluate_model(y_train,train_pred)
    print(f"accuracy:{metrics['accuracy']}")
    print(f"precision:{metrics['precision']}")
    print(f"recall:{metrics['recall']}")
    print(f"F1_score:{metrics['f1_score']}")
    print(f"confusion_metrics:{metrics['confusion_metrics']}")
    
if __name__ == "__main__":
    main()
    


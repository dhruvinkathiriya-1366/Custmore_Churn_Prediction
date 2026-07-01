from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix


def evaluate_model(y_test, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_metrics':confusion_matrix(y_test,y_pred)
     }
    return metrics

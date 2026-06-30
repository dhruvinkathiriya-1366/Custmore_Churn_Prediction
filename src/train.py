# Training Script
# Script to train the churn prediction model

from data_preprocessing import load_data, preprocess_data
from model import ChurnModel

def main():
    """Main training function"""
    # Load data
    train_data = load_data('../data/raw/train.csv')
    
    # Preprocess data
    X_train, y_train = preprocess_data(train_data)
    
    # Train model
    model = ChurnModel()
    model.train(X_train, y_train)
    
    print("Model training completed!")

if __name__ == "__main__":
    main()

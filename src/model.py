# Model Definition Module
# Model architecture and utility functions

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class ChurnModel:
    def __init__(self):
        """Initialize the churn prediction model"""
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)

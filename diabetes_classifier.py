import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Scikit-Learn Ecosystem
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, make_scorer

# Visual Configuration
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class DiabetesDiagnosisSystem:
    """
    A machine learning pipeline to detect diabetes based on clinical features.
    Compares Linear SVM and Multi-Layer Perceptron (Neural Network) performance.
    """
    
    def __init__(self, filepath='data_file.xlsx'):
        self.filepath = filepath
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        
    def load_or_generate_data(self):
        """Loads Excel data if available, otherwise generates synthetic medical data."""
        if os.path.exists(self.filepath):
            print(f"--- Loading real data from {self.filepath} ---")
            raw_data = pd.read_excel(self.filepath)
        else:
            print("--- Warning: Excel file not found. Generating synthetic clinical data... ---")
            # Generate fake data: ID, Gender, Age, BMI, OGTT, HBA1C
            np.random.seed(42)
            n_samples = 500
            
            data = {
                'ID': range(1, n_samples + 1),
                'GENDER': np.random.choice(['Male', 'Female'], n_samples),
                'AGE': np.random.randint(20, 80, n_samples),
                'BMI': np.random.normal(25, 5, n_samples),
                'OGTT1FBS': np.random.normal(110, 30, n_samples), # Glucose
                'HBA1C1': np.random.normal(6.0, 1.5, n_samples)   # Hemoglobin
            }
            raw_data = pd.DataFrame(data)

        # 1. Clean Data
        self.data = raw_data.drop_duplicates().copy()
        print(f"Data Loaded. Shape: {self.data.shape}")

    def preprocess_data(self):
        """Applies clinical thresholds to label data and scales features."""
        print("--- Preprocessing & Labelling ---")
        
        # 1. Labeling based on Medical Thresholds
        # 0 = Non-Diabetic, 1 = Diabetic
        self.data['diabetic'] = 0
        
        # Condition: OGTT >= 126 OR HbA1c >= 6.5 implies Diabetes
        diabetic_mask = (self.data['OGTT1FBS'] >= 126) | (self.data['HBA1C1'] >= 6.5)
        self.data.loc[diabetic_mask, 'diabetic'] = 1
        
        print(f"Class Distribution:\n{self.data['diabetic'].value_counts()}")

        # 2. Encoding Gender
        le = LabelEncoder()
        if 'GENDER' in self.data.columns:
            self.data['GENDER'] = le.fit_transform(self.data['GENDER'])

        # 3. Select Features (First 6 columns as per original logic)
        # Note: Ensure OGTT/HBA1C are inputs only if we are predicting based on them.
        X = self.data.iloc[:, :6]
        y = self.data['diabetic']

        # 4. Split and Scale
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=43, stratify=y
        )
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("Data Split and Scaled.")

    def train_svm(self):
        """Trains and tunes a Linear Support Vector Machine."""
        print("\n--- Training Linear SVM ---")
        
        # Define Grid Search for Hyperparameters
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        scorer = make_scorer(recall_score) # Optimization for Recall (Medical Context)
        
        grid = GridSearchCV(LinearSVC(max_iter=100000, random_state=42), 
                            param_grid, cv=5, scoring=scorer)
        
        grid.fit(self.X_train, self.y_train)
        
        best_svm = grid.best_estimator_
        self.models['SVM'] = best_svm
        print(f"Best SVM Parameters: {grid.best_params_}")
        
        return self.evaluate_model(best_svm, "Linear SVM")

    def train_mlp(self):
        """Trains and tunes a Multi-Layer Perceptron (Neural Network)."""
        print("\n--- Training MLP (Neural Network) ---")
        
        param_grid = {
            'hidden_layer_sizes': [(10,), (50,), (10, 10)],
            'activation': ['tanh', 'relu'],
            'solver': ['adam', 'sgd'],
            'learning_rate': ['constant', 'adaptive']
        }
        
        # Using Recall because missing a diabetic patient is worse than a false alarm
        grid = GridSearchCV(MLPClassifier(max_iter=2000, random_state=42),
                            param_grid, cv=3, scoring='recall', n_jobs=-1)
        
        grid.fit(self.X_train, self.y_train)
        
        best_mlp = grid.best_estimator_
        self.models['MLP'] = best_mlp
        print(f"Best MLP Parameters: {grid.best_params_}")
        
        return self.evaluate_model(best_mlp, "MLP Classifier")

    def evaluate_model(self, model, name):
        """Generic evaluation function with Visualization."""
        preds = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        
        print(f"\nResults for {name}:")
        print(classification_report(self.y_test, preds))
        
        # Plot Confusion Matrix
        cm = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {name}\nAccuracy: {acc:.2%}')
        plt.show()

# --- Main Execution Flow ---
if __name__ == "__main__":
    # Initialize System
    system = DiabetesDiagnosisSystem('data_file.xlsx')
    
    # Run Pipeline
    system.load_or_generate_data()
    system.preprocess_data()
    
    # Train Models
    system.train_svm()
    system.train_mlp()
    

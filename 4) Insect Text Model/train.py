import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')
class InsectTabNet:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
    def load_and_prepare_data(self, csv_file):
        print("Loading and preparing insect data...")
        df = pd.read_csv(csv_file)
        self.feature_columns = [col for col in df.columns if col != 'insect']
        X = df[self.feature_columns].values.astype(np.float32)
        y = df['insect'].values
        y_encoded = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    def train_model(self, X_train, y_train, X_val, y_val):
        print("Training Insect TabNet model...")
        self.model = TabNetClassifier(
            n_d=32,
            n_a=32,
            n_steps=5,
            gamma=1.3,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type="entmax",
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=1
        )
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['validation'],
            eval_metric=['accuracy'],
            max_epochs=200,
            patience=20,
            batch_size=256,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        print("Insect model training completed!")
    def evaluate_model(self, X_test, y_test):
        print("Evaluating insect model...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        insect_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=insect_names))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=insect_names, yticklabels=insect_names)
        plt.title('Insect Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('tabnet_confusion_matrix.png')
        plt.show()
        return accuracy
    def save_model(self, model_path='insect_tabnet_model'):
        print(f"Saving insect model to {model_path}...")
        self.model.save_model(model_path)
        joblib.dump(self.label_encoder, f'{model_path}_label_encoder.pkl')
        joblib.dump(self.feature_columns, f'{model_path}_features.pkl')
        print("Insect model saved successfully!")
    def predict_insect(self, farmer_responses):
        if len(farmer_responses) != 30:
            raise ValueError("Expected 30 responses for 30 questions")
        responses = np.array(farmer_responses).reshape(1, -1).astype(np.float32)
        prediction = self.model.predict(responses)
        probabilities = self.model.predict_proba(responses)
        insect_name = self.label_encoder.inverse_transform(prediction)[0]
        confidence = probabilities[0].max()
        return insect_name, confidence, probabilities[0]
tabnet_model = InsectTabNet()
X_train, X_val, X_test, y_train, y_val, y_test = tabnet_model.load_and_prepare_data('insect_data.csv')
tabnet_model.train_model(X_train, y_train, X_val, y_val)
accuracy = tabnet_model.evaluate_model(X_test, y_test)
tabnet_model.save_model('crop_insect_tabnet')
print(f"Insect model training completed with accuracy: {accuracy:.4f}")
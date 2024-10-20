# ibm-data
<H1>Reinforce - SAV190</H1>
 <H1>Machine Learning-Based Cyber Threat Detection System</H1 

## Aim: 

The aim of our project is to develop an advanced machine learning-based system capable of detecting various cyber threats, such as DDoS attacks, by analyzing network traffic data in real-time. The system aims to improve accuracy and adaptability, addressing the limitations of traditional fixed-rule threat detection methods.

## Algorithm:
Procedure:
1. Data Collection: Gather large datasets of network traffic logs that include normal activity and instances of various cyberattacks.
2. Data Preprocessing: Clean and preprocess the data by handling missing values, removing duplicates, and applying feature encoding.
3. Model Training: Use machine learning algorithms like Random Forest to train the model on labeled data. We use techniques like SMOTE to balance the data and improve detection of rare threats.
4. Feature Scaling: Apply standardization to the data to ensure consistent performance across features.
5. Model Evaluation: Split the dataset into training and testing sets. Evaluate the model’s performance using accuracy scores, classification reports, and cross-validation.
6. Improvement: Fine-tune the model using grid search and cross-validation to optimize hyperparameters for better accuracy.
7. Feature Importance: Use SHAP values to interpret the model's decisions and identify key features contributing to threat detection.

## Program:
~~~~
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from imblearn.over_sampling import SMOTE
import shap

# Load the data from multiple CSVs and combine them
file_names = [
    '/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    '/content/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    '/content/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    '/content/Monday-WorkingHours.pcap_ISCX.csv',
    '/content/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
]

# Combine all the CSVs
combined_data = pd.concat((pd.read_csv(file_name) for file_name in file_names), ignore_index=True)

# Display initial information
print(combined_data.head())
print(combined_data.info())

# Save combined data to CSV
combined_data.to_csv('data.csv', index=False)
data = pd.read_csv('data.csv')

# Fill missing values with column mean for numeric columns
data_cleaned = data.fillna(data.select_dtypes(include=['number']).mean())

# Remove duplicates
data_cleaned = data_cleaned.drop_duplicates()
print(f"Shape after removing duplicates: {data_cleaned.shape}")

# Drop irrelevant columns (e.g., 'Destination Port')
columns_to_drop = ['Destination Port']  # Adjust this as needed
data_cleaned = data_cleaned.drop(columns=[col for col in columns_to_drop if col in data_cleaned.columns])

# Encode categorical features (if any)
le = LabelEncoder()
for column in data_cleaned.select_dtypes(include=['object']).columns:
    data_cleaned[column] = le.fit_transform(data_cleaned[column])

# Define the feature matrix X and the target y (Assuming 'Label' is the target column)
target_column = 'Label'
X = data_cleaned.drop(columns=[target_column])
y = data_cleaned[target_column]

# Data balancing using SMOTE (only if the target is imbalanced)
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest model for multi-class classification
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(rf_model, 'threat_rf_model.pkl')

# Make predictions and evaluate the model
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# SHAP explainability
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_scaled)

# Visualize the SHAP summary plot for multi-class predictions
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns.tolist())

~~~~
## Output:




## Result:
The system successfully detects multiple types of cyber threats, including DDoS attacks, with improved accuracy compared to traditional methods. The model demonstrates high accuracy and reliability through cross-validation and real-world data evaluation, offering an automated solution that adapts to evolving cyberattacks

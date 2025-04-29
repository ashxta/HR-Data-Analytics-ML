# ==========================================
# HR Attrition Full Analysis Pipeline
# ==========================================

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans

from imblearn.over_sampling import SMOTE
import joblib

# 2. Load and Inspect Data
def load_data(path):
    df = pd.read_csv('cleaned_hr_data.csv')
    print(f"Dataset Shape: {df.shape}")
    return df

# 3. Preprocessing Function
def preprocess_data(df):
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    return X, y, label_encoders

# 4. Train/Test Split
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# 5. Apply SMOTE
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_resampled.shape}")
    return X_resampled, y_resampled

# 6. Model Building
def build_model(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

# 7. Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 8. Feature Importance
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feat_df = feat_df.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df.head(15))
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.show()

# 9. Risk Analysis by Department
department_mapping = dict(zip(le.transform(le.classes_), le.classes_))
attrition_by_dept.index = attrition_by_dept.index.map(department_mapping)

# Now plot again
plt.figure(figsize=(6,4))
attrition_by_dept.plot(kind='bar', color='salmon', legend=False)
plt.title('Attrition Risk by Department')
plt.xlabel('Department')
plt.ylabel('Attrition Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 10. Clustering Employees into Risk Groups
def employee_clustering(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=clusters, palette='viridis')
    plt.title('Employee Risk Clustering')
    plt.show()

    return clusters

# 11. Salary Impact Simulation
def salary_simulation(df, model, increase_percent=10):
    df_sim = df.copy()
    original_predictions = model.predict(df_sim.drop('Attrition', axis=1))

    # Simulate salary increase
    if 'MonthlyIncome' in df_sim.columns:
        df_sim['MonthlyIncome'] *= (1 + increase_percent/100)

    new_predictions = model.predict(df_sim.drop('Attrition', axis=1))

    improvement = np.mean(original_predictions) - np.mean(new_predictions)
    print(f"\nAttrition reduction if salary increased by {increase_percent}%: {improvement*100:.2f}%")

# 12. Manager-wise Attrition Risk
def manager_risk_analysis(df):
    if 'ManagerID' in df.columns:
        manager_risk = df.groupby('ManagerID')['Attrition'].mean().sort_values(ascending=False)
        print("\nAttrition Risk by Manager:\n", manager_risk)
        manager_risk.plot(kind='bar', color='lightgreen')
        plt.title('Attrition Risk by Manager')
        plt.ylabel('Attrition Rate')
        plt.show()

# 13. Save Model
def save_model(model, label_encoders, filename='hr_attrition_model_full.pkl'):
    joblib.dump({'model': model, 'encoders': label_encoders}, filename)
    print(f"Model saved to {filename}")

# 14. Main Pipeline
def main():
    # Step 1: Load
    df = load_data('cleaned_hr_data.csv')

    # Step 2: Preprocess
    X, y, label_encoders = preprocess_data(df)

    # Step 3: Train/Test Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # Step 5: Build Model
    model = build_model(X_train_resampled, y_train_resampled)

    # Step 6: Evaluate
    evaluate_model(model, X_test, y_test)

    # Step 7: Feature Importance
    plot_feature_importance(model, X.columns)

    # Step 8: Department Risk
    department_risk_analysis(df)

    # Step 9: Employee Clustering
    clusters = employee_clustering(X)

    # Step 10: Salary Simulation
    salary_simulation(df, model, increase_percent=10)

    # Step 11: Manager-wise Analysis (optional if ManagerID exists)
    if 'ManagerID' in df.columns:
        manager_risk_analysis(df)

    # Step 12: Save model
    save_model(model, label_encoders)

# 15. Execute
if __name__ == "__main__":
    main()

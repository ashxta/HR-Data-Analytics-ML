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
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    return df

# 3. Preprocessing Function
def preprocess_data(df):
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print("\nCategorical columns being encoded:")
    for col in categorical_cols:
        print(f"- {col}")
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
    print("\nClass distribution before SMOTE:")
    print(pd.Series(y_train).value_counts())
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts())
    
    print(f"\nAfter SMOTE: {X_resampled.shape}")
    return X_resampled, y_resampled

# 6. Model Building
def build_model(X_train, y_train):
    print("\nBuilding Random Forest model...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf

# 7. Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Attrition Prediction')
    plt.xlabel('Predicted Attrition Status')
    plt.ylabel('Actual Attrition Status')
    plt.xticks([0.5, 1.5], ['No', 'Yes'])
    plt.yticks([0.5, 1.5], ['No', 'Yes'])
    plt.show()

# 8. Feature Importance
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feat_df = feat_df.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df.head(15))
    plt.title('Top 15 Features Influencing Attrition')
    plt.xlabel('Relative Importance (0-1)')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.show()

# 9. Risk Analysis by Department
def department_risk_analysis(df, label_encoders):
    # Reverse encode Department for interpretability
    if 'Department' in label_encoders:
        df['Department'] = label_encoders['Department'].inverse_transform(df['Department'])
    
    dept_attrition = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False)
    print("\nAttrition Risk by Department:\n", dept_attrition)
    
    plt.figure(figsize=(8, 5))
    dept_attrition.plot(kind='bar', color='salmon')
    plt.title('Attrition Risk by Department')
    plt.xlabel('Department')
    plt.ylabel('Attrition Rate (0-1)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 10. Clustering Employees into Risk Groups
def employee_clustering(X, df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use elbow method to determine optimal clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Cluster Number')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid()
    plt.show()

    # Perform clustering with optimal clusters (3 in this case)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster info to original dataframe
    df['RiskCluster'] = clusters
    
    # Plot clusters using two most important features
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=X_scaled[:, 0], 
        y=X_scaled[:, 1], 
        hue=clusters, 
        palette='viridis',
        alpha=0.7
    )
    plt.title('Employee Risk Clustering')
    plt.xlabel('Standardized Feature 1 (First Principal Component)')
    plt.ylabel('Standardized Feature 2 (Second Principal Component)')
    plt.grid(alpha=0.3)
    plt.show()

    return clusters

# 11. Salary Impact Simulation
def salary_simulation(df, model, label_encoders, increments=[10, 20, 30, 40, 50]):
    df_sim = df.copy()
    
    # Decode Attrition if needed (not used for predictions though)
    if 'Attrition' in label_encoders:
        original_attrition = label_encoders['Attrition'].inverse_transform(df_sim['Attrition'])

    X_base = df_sim.drop('Attrition', axis=1)
    original_predictions = model.predict_proba(X_base)[:, 1]
    base_rate = np.mean(original_predictions) * 100

    avg_attrition_rates = [base_rate]  # Start with current attrition rate
    labels = ["Current"]

    for inc in increments:
        df_mod = df_sim.copy()
        if 'MonthlyIncome' in df_mod.columns:
            df_mod['MonthlyIncome'] *= (1 + inc / 100)
        
        X_mod = df_mod.drop('Attrition', axis=1)
        predictions = model.predict_proba(X_mod)[:, 1]
        avg_attrition = np.mean(predictions) * 100
        avg_attrition_rates.append(avg_attrition)
        labels.append(f"+{inc}%")

        print(f"→ Attrition if salary increased by {inc}%: {avg_attrition:.2f}%")

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, avg_attrition_rates, color=['red'] + ['green'] * len(increments))
    plt.title("Impact of Salary Increase on Predicted Attrition")
    plt.xlabel("Salary Increase (%)")
    plt.ylabel("Avg Predicted Attrition Rate (%)")
    plt.ylim(0, max(avg_attrition_rates) + 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# 12. Manager-wise Attrition Risk
def manager_risk_analysis(df, label_encoders):
    if 'YearsWithCurrManager' in df.columns:
        # Create manager tenure groups
        df['ManagerTenureGroup'] = pd.cut(
            df['YearsWithCurrManager'],
            bins=[0, 2, 5, 10, 50],
            labels=['0-2 years', '2-5 years', '5-10 years', '10+ years']
        )
        
        # Calculate attrition by manager tenure
        tenure_attrition = df.groupby('ManagerTenureGroup', observed = 'True')['Attrition'].mean()
        
        plt.figure(figsize=(8, 5))
        tenure_attrition.plot(kind='bar', color='lightgreen')
        plt.title('Attrition Risk by Manager Tenure')
        plt.xlabel('Years with Current Manager')
        plt.ylabel('Attrition Rate')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

# 13. Work-Life Balance Analysis
def work_life_analysis(df, label_encoders):
    if 'WorkLifeBalance' in df.columns:
        # Reverse encode Attrition for interpretability
        if 'Attrition' in label_encoders:
            df['Attrition'] = label_encoders['Attrition'].inverse_transform(df['Attrition'])
        
        plt.figure(figsize=(10, 6))
        sns.countplot(
            x='WorkLifeBalance',
            hue='Attrition',
            data=df,
            palette='coolwarm'
        )
        plt.title('Attrition by Work-Life Balance Rating')
        plt.xlabel('Work-Life Balance Rating (1-4)')
        plt.ylabel('Number of Employees')
        plt.legend(title='Attrition Status')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

# 14. Save Model
def save_model(model, label_encoders, filename='hr_attrition_model_enhanced.pkl'):
    joblib.dump({'model': model, 'encoders': label_encoders}, filename)
    print(f"\nModel and encoders saved to {filename}")

# 15. Main Pipeline
def main():
    # Step 1: Load
    print("="*50)
    print("STEP 1: LOADING DATA")
    print("="*50)
    df = load_data('cleaned_hr_data.csv')
    
    # Step 2: Preprocess
    print("\n" + "="*50)
    print("STEP 2: DATA PREPROCESSING")
    print("="*50)
    X, y, label_encoders = preprocess_data(df)
    
    # Step 3: Train/Test Split
    print("\n" + "="*50)
    print("STEP 3: TRAIN/TEST SPLIT")
    print("="*50)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: SMOTE
    print("\n" + "="*50)
    print("STEP 4: APPLYING SMOTE")
    print("="*50)
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Step 5: Build Model
    print("\n" + "="*50)
    print("STEP 5: MODEL TRAINING")
    print("="*50)
    model = build_model(X_train_resampled, y_train_resampled)
    
    # Step 6: Evaluate
    print("\n" + "="*50)
    print("STEP 6: MODEL EVALUATION")
    print("="*50)
    evaluate_model(model, X_test, y_test)
    
    # Step 7: Feature Importance
    print("\n" + "="*50)
    print("STEP 7: FEATURE IMPORTANCE")
    print("="*50)
    plot_feature_importance(model, X.columns)
    
    # Step 8: Department Risk
    print("\n" + "="*50)
    print("STEP 8: DEPARTMENT RISK ANALYSIS")
    print("="*50)
    department_risk_analysis(df.copy(), label_encoders)
    
    # Step 9: Employee Clustering
    print("\n" + "="*50)
    print("STEP 9: EMPLOYEE RISK CLUSTERING")
    print("="*50)
    clusters = employee_clustering(X, df.copy())
    
    # Step 10: Salary Simulation
    print("\n" + "="*50)
    print("STEP 10: SALARY IMPACT SIMULATION")
    print("="*50)
    salary_simulation(df.copy(), model, label_encoders, increments=[10, 20, 30, 40, 50])

    
    # Step 11: Manager-wise Analysis
    print("\n" + "="*50)
    print("STEP 11: MANAGER TENURE ANALYSIS")
    print("="*50)
    manager_risk_analysis(df.copy(), label_encoders)
    
    # Step 12: Work-Life Balance Analysis
    print("\n" + "="*50)
    print("STEP 12: WORK-LIFE BALANCE ANALYSIS")
    print("="*50)
    work_life_analysis(df.copy(), label_encoders)
    
    # Step 13: Save model
    print("\n" + "="*50)
    print("STEP 13: SAVING MODEL")
    print("="*50)
    save_model(model, label_encoders)

# 16. Execute
if __name__ == "__main__":
    main()

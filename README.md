# HR Attrition Analysis and Prediction System

## 📚 Overview

This project aims to not just predict employee attrition but also provide **deep actionable business insights** to the HR department.  
We built an end-to-end machine learning pipeline that predicts attrition risk, explains key contributing factors, simulates salary adjustments, analyzes department/manager risk areas, and clusters employees into strategic risk groups for targeted retention.

---

## 🛠️ Tools & Technologies Used

- **Python 3**
- **Pandas, NumPy** (Data Processing)
- **Matplotlib, Seaborn** (Visualization)
- **scikit-learn** (Modeling, Preprocessing)
- **imbalanced-learn** (SMOTE for balancing dataset)
- **KMeans Clustering** (Risk grouping)
- **Joblib** (Model Saving)
- **GridSearchCV** (Hyperparameter tuning)

---

## 🎯 Key Features

| Feature | Description |
|:--------|:------------|
| 🎯 Attrition Prediction | Predicts whether an employee is likely to leave or stay |
| 🔍 Feature Importance Analysis | Identifies top drivers of employee attrition |
| 🏢 Department Risk Analysis | Reveals departments with the highest attrition risk |
| 🧑‍💼 Manager-wise Attrition Scoring | Identifies managers with high employee turnover rates |
| 💰 Salary Adjustment Simulation | Quantifies the impact of salary hikes on attrition rates |
| 🧩 Employee Clustering | Segments employees into risk groups for customized retention strategies |
| 📦 Model Deployment Ready | Trained model and encoders saved using Joblib for easy integration |

---

## 📊 Business Impact

✅ HR leaders can focus on **departments** and **managers** with highest risks.  
✅ **Simulated salary strategies** can optimize compensation budgets.  
✅ **Risk clustering** enables **targeted employee retention programs** (bonuses, job role changes, work flexibility).

---

## 🔥 Unique Additions

- **Salary Impact Simulation**: Shows how small salary hikes could lower attrition.
- **Employee Risk Grouping (via Clustering)**: Not all at-risk employees are the same — we created strategic groups.
- **Manager-wise Performance**: Pinpoint people managers causing excessive attrition.
- **Actionable Recommendations**: Not just technical modeling, but true business advisory outputs.

---

## 📈 Workflow Diagram

flowchart TD
    A[Data Collection] --> B[Data Preprocessing]
    B --> C[Model Building (Random Forest)]
    C --> D[Evaluation & Feature Importance]
    D --> E[SMOTE Oversampling]
    D --> F[Department Risk Analysis]
    D --> G[Salary Impact Simulation]
    D --> H[Employee Risk Clustering]
    E --> I[Final Model Saving]
    G --> I
    F --> I
    H --> I

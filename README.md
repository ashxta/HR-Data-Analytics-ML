# HR Attrition Analysis and Prediction System

## 📚 Overview

This project builds a complete HR analytics pipeline that not only predicts employee attrition but also generates deep, interpretable, and actionable business insights for HR decision-making.
 Developed an end-to-end system that:

-> Predicts attrition risk using a trained ML model

-> Explains key contributing factors

-> Simulates salary hikes for targeted retention

-> Analyzes risk at the department and manager level

-> Clusters employees into risk groups for strategic interventions

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
|--------|-------------|
| 🔍 **Attrition Prediction** | Uses a Random Forest model to predict whether an employee is likely to leave |
| 📊 **Feature Importance with Summary** | Prints and plots the top features influencing attrition |
| 🏢 **Department Risk Analysis** | Outputs departments sorted by historical attrition risk |
| 🧑‍💼 **Manager-wise Risk Breakdown** | Reports attrition by tenure with current manager |
| 💰 **Salary Adjustment Simulation (Targeted)** | Simulates attrition reduction when giving raises to lower-income employees |
| 🧩 **Employee Clustering with Risk Scores** | Groups employees into 3 risk clusters and reports predicted attrition per group |
| 📈 **Clear Printed Summaries** | Every major step prints useful business-level insights |
| 💾 **Model Saving** | Saves trained model + label encoders for deployment |

---

## 📊 Business Impact

✅ HR leaders can identify departments, managers, and income segments that drive attrition
✅ Salary strategy simulations help optimize retention budgeting
✅ Cluster-based risk segmentation supports personalized retention strategies
✅ Printed insights bring the “why” behind attrition — not just “what”

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
  B --> C[SMOTE Oversampling]
  C --> D[Model Building (Random Forest)]
  D --> E[Model Evaluation & Feature Importance]
  E --> F[Department Risk Analysis]
  E --> G[Salary Impact Simulation (Targeted)]
  E --> H[Employee Clustering (Risk-Based)]
  F --> I[Manager & Work-Life Analysis]
  G --> I
  H --> I
  I --> J[Final Model Saving]


🧠 HR Attrition Analysis and Prediction System
📚 Overview
This project builds a complete HR analytics pipeline that not only predicts employee attrition but also generates deep, interpretable, and actionable business insights for HR decision-making.

We developed an end-to-end system that:

Predicts attrition risk using a trained ML model

Explains key contributing factors

Simulates salary hikes for targeted retention

Analyzes risk at the department and manager level

Clusters employees into risk groups for strategic interventions

🛠️ Tools & Technologies Used
Python 3

Pandas, NumPy – Data manipulation

Matplotlib, Seaborn – Visual & statistical plots

scikit-learn – Machine Learning (Random Forest)

imbalanced-learn – SMOTE (for class imbalance)

KMeans – Employee clustering

Joblib – Model persistence

GridSearchCV – Hyperparameter tuning

🎯 Key Features

Feature	Description
🔍 Attrition Prediction	Uses a Random Forest model to predict whether an employee is likely to leave
📊 Feature Importance with Summary	Prints and plots the top features influencing attrition
🏢 Department Risk Analysis	Outputs departments sorted by historical attrition risk
🧑‍💼 Manager-wise Risk Breakdown	Reports attrition by tenure with current manager
💰 Salary Adjustment Simulation (Targeted)	Simulates attrition reduction when giving raises to lower-income employees
🧩 Employee Clustering with Risk Scores	Groups employees into 3 risk clusters and reports predicted attrition per group
📈 Clear Printed Summaries	Every major step prints useful business-level insights
💾 Model Saving	Saves trained model + label encoders for deployment
📦 Business Impact
✅ HR leaders can identify departments, managers, and income segments that drive attrition
✅ Salary strategy simulations help optimize retention budgeting
✅ Cluster-based risk segmentation supports personalized retention strategies
✅ Printed insights bring the “why” behind attrition — not just “what”

🔥 Unique Additions
💰 Targeted Salary Simulation: Models how raises for lower-paid staff could impact predicted attrition

🧩 Risk Clustering: Not all at-risk employees are the same — cluster analysis groups them strategically

📋 Step-by-step Printed Summaries: Key insights at each stage (feature importance, manager analysis, etc.)

🧠 True Business Insights, Not Just Code: Outputs help drive HR policy

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


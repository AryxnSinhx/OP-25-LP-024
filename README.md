🧠 Mental Wellness Analysis and Support Strategy
Project.

📌 Objective
This project analyzes mental health in the tech industry using machine learning to:
• Classification → Predict if a person is likely to seek mental health treatment.
• Regression → Predict an individual’s age from workplace and personal attributes.
• Unsupervised Clustering → Segment employees into personas (e.g., Silent Strugglers, Supported Advocates).
An interactive Streamlit dashboard is deployed for HR teams to explore insights, simulate interventions, and flag high-risk employee groups.


📊 Dataset
• Source: Mental Health in Tech Survey (OSMI)
• Size: ~1,500 tech professionals
• Features: 
• Demographics (age, gender, country)
• Workplace environment (mental health benefits, leave policies)
• Personal experiences (mental illness, family history)
• Attitudes towards mental health


🔎 Project Workflow
1. Exploratory Data Analysis (EDA)
• Data cleaning (removing invalid ages, gender anomalies)
• Visualizations (distribution plots, correlation heatmaps, workplace policy insights)
2. Supervised Learning
• Classification Models: Logistic Regression, Random Forest, XGBoost, SVM
• Regression Models: Linear Regression, Random Forest Regressor
• Evaluation Metrics: Accuracy, ROC-AUC, RMSE, MAE
3. Unsupervised Learning
• Algorithms: KMeans, Agglomerative Clustering, DBSCAN
• Evaluation: Silhouette Score
• Personas identified (e.g., Silent Sufferers, Open Advocates)
4. Deployment with Streamlit
• EDA Visualizations
• Prediction Form (seek treatment, predict age)
• Cluster Visualizer with persona descriptions

The link to the WebApp:
https://op-25-lp-024-jnyuuga8a3mxzpvxwvi8zr.streamlit.app/
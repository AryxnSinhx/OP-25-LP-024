import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Mental Wellness Dashboard", layout="wide")

# ========================
# 📌 Project Overview
# ========================
st.title("🧠 Mental Wellness Analysis & Support Strategy")
st.subheader("Data-Driven Insights for Mental Health in the Tech Workforce")

st.markdown("""
### Objective
To understand the key factors influencing mental health in the tech industry and provide strategic, data-driven solutions:

- 🔍 **Classification Task:** Predict if someone is likely to seek mental health treatment  
- 📈 **Regression Task:** Predict an individual's age based on workplace and personal traits  
- 🧩 **Clustering Task:** Segment employees into mental health personas for targeted HR strategies

---

### 📊 Dataset Overview
- **Source:** Mental Health in Tech Survey by OSMI
- **Size:** 1,500+ respondents from the tech industry
- **Features Include:**
  - Demographics (Age, Gender, Country)
  - Workplace Policies (Mental health benefits, Leave policies)
  - Mental Health Experiences (History, Treatment, Openness)

---

### 👨‍💻 Case Study Background
As a Machine Learning Engineer at **NeuronInsights Analytics**, you’ve been contracted by a tech coalition (CodeLab, QuantumEdge, SynapseWorks) to:

> **“Analyze burnout, disengagement, and attrition trends tied to mental wellness.”**

They seek your expertise to:
- Identify employees at risk of silent suffering
- Understand how remote work, HR support, and mental health benefits influence well-being
- Enable **HR leaders** to simulate interventions and monitor mental wellness — anonymously but impactfully

---

### 🧠 Your Deliverables:
- **Part 1: EDA**
    - Clean, visualize, and analyze core patterns
- **Part 2: Supervised ML**
    - Classification: Predict mental health treatment seeking
    - Regression: Predict age for behavioral insight
- **Part 3: Unsupervised ML**
    - Cluster employees into profiles like:
        - “Silent Sufferers”
        - “Open Advocates”
        - “Under-Supported Professionals”

---

### 💡 Purpose
To bridge machine learning with mental health awareness — transforming numbers into **actionable empathy**.
""")


# Load Data
df_clean = pd.read_csv(r"Project/files_app/Survey Reprised 2.0.csv")
df_raw = pd.read_csv(r"Project/files_app/survey.csv")

# Display first five rows of cleaned dataset
st.subheader("🔍 Sample of Cleaned Dataset")
st.dataframe(df_clean.head())

def show_insight(text):
    with st.expander("📌 Show Insight"):
        st.markdown(text)

# ========== Section 1: Demographics ==========
st.header("📊 Overview and Missing Data")
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Age Distribution")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(df_clean['Age'], bins=30, kde=True, ax=ax)
    st.pyplot(fig, use_container_width=True)
    show_insight("Most participants are between 20 and 35. Some age outliers exist.")

with col2:
    st.subheader("2. Missing Data Contribution (Raw Dataset)")
    missing_raw = df_raw.isnull().sum()
    missing_raw = missing_raw[missing_raw > 0]
    total_missing = missing_raw.sum()
    missing_percent = (missing_raw / total_missing) * 100
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(missing_percent, labels=missing_percent.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig, use_container_width=True)
    show_insight("Most missing data comes from 'comments', followed by 'state' and 'work_interfere'.")

# ========== Section 2: Gender Views ==========
st.header("👩‍💼 Gender-Based Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("3. Seek Help by Gender")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(data=df_clean, x="seek_help", hue="Gender_cleaned", ax=ax)
    st.pyplot(fig, use_container_width=True)
    show_insight("Females are more likely to seek help compared to males.")

with col2:
    st.subheader("4. Mental Health Interview by Gender")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(data=df_clean, x="mental_health_interview", hue="Gender_cleaned", ax=ax)
    st.pyplot(fig, use_container_width=True)
    show_insight("Slightly more females report being open to discussing mental health in interviews.")

# ========== Section 3: Tech Company vs Policies ==========
st.header("🏢 Tech Company and HR Policies")
hues = ['wellness_program', 'care_options', 'mental_health_consequence']
cols = st.columns(3)

for i, h in enumerate(hues):
    with cols[i]:
        st.subheader(f"{i+5}. Tech vs {h.replace('_', ' ').title()}")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(data=df_clean, x='tech_company', hue=h, palette='Set2', ax=ax)
        ax.set_title(f'Tech Company vs {h.replace("_", " ").title()}', fontsize=10)
        ax.set_xlabel('Tech Company')
        st.pyplot(fig, use_container_width=True)
        show_insight(f"Tech companies show variation in support for {h.replace('_', ' ')}.")

# ========== Section 4: Workplace Support ==========
st.header("🧑‍💼 Workplace Consequences")

col1, col2 = st.columns(2)

with col1:
    st.subheader("8. Supervisor vs Mental Health Consequence")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(data=df_clean, x="supervisor", hue="mental_health_consequence", ax=ax)
    st.pyplot(fig, use_container_width=True)
    show_insight("Supportive supervisors link to fewer reported mental health consequences.")

with col2:
    st.subheader("9. Leave Policy vs Mental Health Consequence")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(data=df_clean, x="leave", hue="mental_health_consequence", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    st.pyplot(fig, use_container_width=True)
    show_insight("Stricter or unclear leave policies correlate with more negative consequences.")

# ========== Section 5: Donut and Heatmap ==========
st.header("📈 Donut & Correlation Heatmap")

col1, col2 = st.columns(2)

with col1:
    st.subheader("10. Donut Chart: Seek Help")
    seek_counts = df_clean['seek_help'].value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(seek_counts, labels=seek_counts.index, autopct='%1.1f%%', startangle=90,
           wedgeprops=dict(width=0.4))
    ax.axis('equal')
    st.pyplot(fig, use_container_width=True)
    show_insight("A considerable number of individuals still do not seek help.")

with col2:
    st.subheader("11. Heatmap: Work Interference vs Treatment")
    heat_df = df_clean[['work_interfere', 'treatment']].dropna().copy()
    heat_df['work_interfere'] = heat_df['work_interfere'].astype('category').cat.codes
    heat_df['treatment'] = heat_df['treatment'].astype('category').cat.codes
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(heat_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig, use_container_width=True)
    show_insight("Higher work interference correlates with a greater likelihood of treatment.")

# Footer
st.markdown("---")

st.markdown("📊 Dashboard built with ❤️ using Streamlit.")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Set up page
st.set_page_config(page_title="Clustering Insights", layout="wide")
st.title("🧩 Clustering Mental Health Personas")

# =======================
# Overview of Clustering
# =======================
st.markdown("""
## 🧩 Overview: Unsupervised Learning with KMeans Clustering

In this section, we use **KMeans Clustering**, an unsupervised machine learning algorithm, to identify distinct mental health personas based on workplace support, prior experience, and willingness to seek help.

**KMeans Clustering** works by:
- Grouping similar observations into clusters based on feature similarity
- Minimizing the variance within each cluster
- Assigning each respondent to the nearest cluster center

### 🎯 Goal:
To segment employees into **3 key clusters** that can guide HR policies and mental health interventions.

We applied t-SNE (a dimensionality reduction method) to visually inspect how these clusters separate in two dimensions.
""")

# =======================
# Hidden Step 1: Preprocessing
# =======================
df = pd.read_csv(r'files_app\Survey Reprised 2.0.csv')
df_filtered = df[(df['Age'] >= 21) & (df['Age'] <= 45)]

features = [
    'care_options', 'benefits', 'coworkers',
    'supervisor', 'family_history', 'treatment', 'work_interfere'
]
df_cluster = df_filtered[features].copy()

# Encode categorical features
label_encoders = {}
for col in df_cluster.columns:
    le = LabelEncoder()
    df_cluster[col] = le.fit_transform(df_cluster[col].astype(str))
    label_encoders[col] = le

# Perform clustering with fixed K=3
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans_final.fit_predict(df_cluster)

# =======================
# Step 1: Visualizations
# =======================
st.markdown("### 📊 Clustering Results")

col1, col2 = st.columns(2)

# Left: t-SNE Scatter Plot
with col1:
    st.markdown("#### 🧬 t-SNE Cluster Scatter Plot")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(df_cluster.drop(columns='Cluster'))

        df_tsne = pd.DataFrame(tsne_result, columns=['tSNE1', 'tSNE2'])
        df_tsne['Cluster'] = df_cluster['Cluster'].values

        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df_tsne, x='tSNE1', y='tSNE2', hue='Cluster', palette='Set2', ax=ax2)
        ax2.set_title("t-SNE: KMeans Clustering (K=3)")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"t-SNE visualization failed: {e}")

# Right: Cluster Count Plot
with col2:
    st.markdown("#### 📊 Cluster Sizes")
    cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
    fig3, ax3 = plt.subplots()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='Set2', ax=ax3)
    ax3.set_title("Number of Respondents in Each Cluster")
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)

# =======================
# Step 2: Cluster Interpretations
# =======================
st.markdown("### 🧠 Cluster Interpretations")

st.markdown("#### 🔵 Cluster 0")
st.markdown("""
- May have **limited workplace support**
- **Less open** about mental health
- Experience **higher work interference**
""")

st.markdown("#### 🟢 Cluster 1")
st.markdown("""
- **Likely to seek treatment**
- Have access to **care options and benefits**
- Comfortable discussing issues with **supervisors**
""")

st.markdown("#### 🟣 Cluster 2")
st.markdown("""
- Possibly **unaware or unsupported**
- Show **mixed interaction** with coworkers and HR
- May fall into a **gray zone** between needing help and getting it
""")

# =======================
# Step 3: View Full Data
# =======================
with st.expander("📄 Show Full Clustered Dataset"):
    st.dataframe(df_cluster)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by NeuronInsights")

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---- PAGE SETUP ----
st.set_page_config(page_title="Course Recommender", layout="wide")

# ---- STYLING ----
st.markdown("""
    <style>
        .main {background-color: #f4f6f9;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .title-style {
            font-size: 36px;
            font-weight: 800;
            color: #2c3e50;
        }
        .card {
            padding: 20px;
            margin-bottom: 15px;
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .recommend-title {
            font-size: 20px;
            font-weight: bold;
            color: #34495e;
        }
    </style>
""", unsafe_allow_html=True)

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_excel("online_course_recommendation_v2.xlsx")
    df = df[['user_id', 'course_name', 'instructor', 'rating','course_duration_hours']]
    df.dropna(inplace=True)
    return df

df = load_data()

# ---- HEADER ----
st.markdown('<h1 class="title-style">🎓 User-Based Course Recommender on Rating</h1>', unsafe_allow_html=True)
st.markdown("**📢 Choose a user to get smart course recommendations based on similar learners.**")

# ---- FILTERING ACTIVE USERS ----
user_counts = df['user_id'].value_counts()
active_users = user_counts[user_counts >= 5].index[:1000]
df_sampled = df[df['user_id'].isin(active_users)]

# ---- SIDEBAR ----
st.sidebar.header("🔍 User Settings")
user_ids = df_sampled['user_id'].unique()
selected_user = st.sidebar.selectbox("Select User ID", user_ids)
top_n = st.sidebar.slider("Number of Recommendations", 1,10,20, 1)

# ---- BUTTON ----
if st.sidebar.button("✨ Get Recommendations"):
    user_course_matrix = df_sampled.pivot_table(index='user_id', columns='course_name', values='rating')
    user_course_matrix.fillna(0, inplace=True)

    similarity = cosine_similarity(user_course_matrix)
    sim_df = pd.DataFrame(similarity, index=user_course_matrix.index, columns=user_course_matrix.index)
    similar_users = sim_df[selected_user].sort_values(ascending=False)[1:6]

    unrated = user_course_matrix.loc[selected_user][user_course_matrix.loc[selected_user] == 0].index
    sim_scores = pd.Series(0, index=unrated)

    for course in unrated:
        ratings = user_course_matrix.loc[similar_users.index, course]
        weights = similar_users
        if ratings[ratings > 0].any():
            sim_scores[course] = (ratings * weights).sum() / weights[ratings > 0].sum()

    sim_scores = sim_scores.dropna().sort_values(ascending=False).head(top_n)
    recommended = df[df['course_name'].isin(sim_scores.index)].drop_duplicates('course_name')

    if not recommended.empty:
        st.success(f"✅ Top {top_n} Personalized Recommendations for **User {selected_user}**")

        for _, row in recommended.iterrows():
            st.markdown(f"""
                <div class="card">
                    <div class="recommend-title">📘 {row['course_name']}</div>
                    <div>👨‍🏫 <strong>Instructor:</strong> {row['instructor']}</div>
                    <div>⭐ <strong>Predicted Rating:</strong> {sim_scores[row['course_name']]:.2f}</div>
                    <div>⏱️ <strong>Duration:</strong> {row['course_duration_hours']} hours</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No new recommendations found for this user.")
else:
    st.info("ℹ️ Select a User ID and click *Get Recommendations*.")

with st.expander("ℹ️ What does Predicted Rating mean?"):
    st.markdown("""
        The **Predicted Rating** is estimated using **User-Based Collaborative Filtering**.

        - We identify users with similar course preferences (using **cosine similarity**).
        - Then, we compute a **weighted average** of their ratings for courses you haven't rated.
        - The more similar the user, the more weight their opinion has.

        📌 This helps us estimate how likely you are to **enjoy a course**, even if you haven't seen it before.
    """)




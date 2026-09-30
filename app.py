import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI-Powered Product Review Intelligence System", layout="wide")

# Custom CSS to force vibrant green styling on ALL primary Streamlit buttons
st.markdown("""

""", unsafe_allow_html=True)

# Load Model & Vectorizer
@st.cache_resource
def load_assets():
    with open("models/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_assets()

st.title("AI-Powered Product Review Intelligence System")
st.write("Enter a product review to predict its sentiment.")

# Metric Summary Cards
st.subheader("Dashboard Analytics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", "154,475")
col2.metric("Positive Reviews", "122,813")
col3.metric("Negative Reviews", "23,353")
col4.metric("Neutral Reviews", "8,309")

st.divider()

# Navigation Tabs
tab1, tab2 = st.tabs(["Single Review Sentiment Analyzer", "Batch CSV Review Analysis"])

with tab1:
    st.subheader("Analyze Single Review")
    review_input = st.text_area("Enter your review:", placeholder="The product quality is excellent and delivery was fast...")
    
    if st.button("Analyze Sentiment", key="single_analyze"):
        if review_input.strip():
            # Preprocess & Vectorize
            review_vector = tfidf.transform([review_input.lower()])
            prediction = model.predict(review_vector)[0]
            
            # Display Prediction
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(review_vector)[0]
                confidence = max(probs) * 100
                st.success(f"Predicted Sentiment: **{prediction.upper()}** (Confidence: {confidence:.2f}%)")
            else:
                st.success(f"Predicted Sentiment: **{prediction.upper()}**")
        else:
            st.warning("Please enter a review to analyze.")

with tab2:
    st.subheader("Batch Review Analysis")
    uploaded_file = st.file_uploader("Upload CSV containing a 'Review' column", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if "Review" in batch_df.columns:
            clean_reviews = batch_df["Review"].astype(str).str.lower()
            batch_vectors = tfidf.transform(clean_reviews)
            batch_df["Predicted_Sentiment"] = model.predict(batch_vectors)
            
            st.write("### Predictions Output", batch_df.head(10))
            
            # Chart
            st.write("### Sentiment Distribution")
            counts = batch_df["Predicted_Sentiment"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3))
            counts.plot(kind="bar", ax=ax, color=["#008a00", "#d62728", "#1f77b4"])
            plt.ylabel("Number of Reviews")
            st.pyplot(fig)
            
            # Download Button
            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", data=csv_data, file_name="analyzed_reviews.csv", mime="text/csv")
        else:
            st.error("The uploaded CSV file must contain a 'Review' column.")
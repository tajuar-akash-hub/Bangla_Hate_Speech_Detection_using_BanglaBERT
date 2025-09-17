import streamlit as st
import requests

# Page config
st.set_page_config(page_title="Bangla Hate Speech Detector", page_icon="🔍")

# Title
st.title("🔍 Bangla Hate Speech Detector")

# Text input
text = st.text_area("Enter Bangla text:", placeholder="এখানে আপনার বাংলা টেক্সট লিখুন...")

# Analyze button
if st.button("🔍 Analyze", type="primary") and text.strip():
    try:
        # Call API
        response = requests.post(
            "http://localhost:8000/predict",
            json={"text": text},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Show result
            prediction = result['prediction']
            confidence = result['confidence']
            
            if prediction == "Hate":
                st.error(f"🔴 **Hate Speech Detected** (Confidence: {confidence:.1%})")
            else:
                st.success(f"🟢 **No Hate Speech** (Confidence: {confidence:.1%})")
                
        else:
            st.error("API Error")
            
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to API. Make sure FastAPI server is running on localhost:8000")
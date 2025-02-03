import streamlit as st
import pickle

# Load trained model and vectorizer
with open("spam.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tvf.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("📧 Spam Mail Classifier")
st.write("Enter an email message to check whether it's Spam or Not Spam.")

# User Input
email_text = st.text_area("Enter Email Text Here:", height=150)

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter an email text for prediction.")
    else:
        # Transform input text
        email_vector = vectorizer.transform([email_text])
        
        # Predict using trained model
        prediction = model.predict(email_vector)[0]
        
        # Show result
        if prediction == 1:
            st.error("🚨 This is a Spam Email!")
        else:
            st.success("✅ This is NOT a Spam Email.")

st.markdown("Developed by **Hardik Chudasama** | Contact: chudasamahardik333@gmail.com")
# Run the app with: streamlit run filename.py

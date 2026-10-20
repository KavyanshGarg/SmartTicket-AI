
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# App title
st.title("SmartTicket AI")
st.write("NLP-Based Customer Support Ticket Classifier")

# Training dataset

data = {
    "ticket": [
        # Technical
        "My computer is not working",
        "The application keeps crashing",
        "My software is showing an error",
        "The website is loading very slowly",
        "I cannot upload my file",
        "The app freezes when I open it",
        "I am getting a server error",
        "The page is not displaying correctly",
        "The website is down",
        "The app is not responding",
        "I get an error when saving changes",
        "The dashboard does not load",
        "The download keeps failing",
        "I cannot connect to the server",
        "The application has a technical bug",
        "The page shows a blank screen",

        # Account
        "I cannot log into my account",
        "Please reset my password",
        "My account has been locked",
        "I forgot my username",
        "I cannot verify my email",
        "Please help me change my password",
        "My account login is not working",
        "I need to update my profile",
        "I cannot access my profile",
        "My password is not accepted",
        "I want to change my email address",
        "My account verification failed",
        "I am unable to sign in",
        "Please unlock my account",
        "I forgot my login details",
        "My profile information is incorrect",

        # Billing
        "I was charged twice",
        "I need a refund for my payment",
        "My invoice has an incorrect amount",
        "My payment was declined",
        "I want to cancel my subscription",
        "I have a question about my bill",
        "My refund has not arrived",
        "I was charged for a cancelled plan",
        "There is an unexpected charge",
        "My subscription payment failed",
        "Please explain this invoice",
        "I need a copy of my receipt",
        "I want to update my billing details",
        "My card was charged incorrectly",
        "I was billed for the wrong plan",
        "When will my refund be processed",

        # General
        "How can I contact customer support",
        "I want information about your services",
        "Tell me your business hours",
        "Where can I find your documentation",
        "I want to know about your company",
        "How do I contact your team",
        "Can you explain your services",
        "Where can I find the user guide",
        "Do you offer any training",
        "Where can I read the FAQ",
        "How can I submit feedback",
        "I need general product information",
        "Where is your help center",
        "Can I speak to a representative",
        "What features does your product offer",
        "Where can I find the documentation"
    ],
    "category": (
        ["Technical"] * 16
        + ["Account"] * 16
        + ["Billing"] * 16
        + ["General"] * 16
    )
}


df = pd.DataFrame(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["ticket"],
    df["category"],
    test_size=0.25,
    random_state=42,
    stratify=df["category"]
)

# Create NLP machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display evaluation results
st.subheader("Model Evaluation")
st.metric("Test Accuracy", f"{accuracy:.0%}")
st.caption(
    "Preliminary result based on only 8 held-out examples. "
    "It may not reflect real-world performance."
)

# Optional classification report
with st.expander("View Classification Report"):
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose())

# Ticket classification
st.subheader("Classify a Support Ticket")

ticket = st.text_area("Enter your support ticket:")

if st.button("Classify Ticket"):
    if ticket.strip():
        prediction = model.predict([ticket])[0]
        probabilities = model.predict_proba([ticket])[0]
        confidence = max(probabilities) * 100

        st.success(f"Predicted Category: {prediction}")
        st.write(f"Model Confidence: {confidence:.2f}%")
        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee of correctness."
        )
    else:
        st.warning("Please enter a support ticket.")

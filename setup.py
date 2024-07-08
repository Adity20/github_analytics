import streamlit as st
from transformers import pipeline

# Initialize the Gemini model pipeline for question-answering
gemini_qa_pipeline = pipeline("question-answering", model="microsoft/CodeGPT-small-py", tokenizer="microsoft/CodeGPT-small-py")

def provide_recommendation(issue_description):
    # Create a prompt for the Gemini model
    question = "How can this issue be resolved?"
    answer = gemini_qa_pipeline({
        'context': issue_description,
        'question': question
    })
    return answer['answer']

# Streamlit app layout
st.title("GitHub Issue Solution Recommender")
st.markdown("""
    Enter a GitHub issue description to receive a recommended solution.
""")

issue_description = st.text_area("Issue Description", height=300)

if st.button("Get Recommendation"):
    if issue_description.strip():
        with st.spinner('Generating recommendation...'):
            recommendation = provide_recommendation(issue_description)
            st.subheader("Recommended Solution")
            st.write(recommendation)
    else:
        st.error("Please enter an issue description.")

import streamlit as st
import openai
import json
from IPython.display import HTML, display
from process_img_response import is_base64
from create_rag import retriever
from openai import OpenAI

# Set your OpenAI API key here
client = OpenAI(api_key="")

def call_gpt4o(system_prompt, user_prompt_text):

    messages = [{"role": "developer", "content": system_prompt}]

    # Add user data (already a string)
    messages.append({"role": "user", "content": user_prompt_text})

    # Call GPT-4o API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.01,  # Reduce randomness for factual accuracy
        top_p=0.01         # Reduce sampling to avoid hallucinations
    )

    # Extract and clean response
    output_text = response.choices[0].message.content

    return output_text

# Function to call the RAG pipeline (integration with OpenAI GPT-4)
def call_rag(user_prompt_text):
    # Generate a response based on the user query using GPT-4
    response = openai.Completion.create(
        engine="gpt-4",  # Using GPT-4 model
        prompt=f"{system_prompt}\nUser query: {user_prompt_text}\nResponse:",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Streamlit app layout
st.title("Chatbot")
st.write("Ask me anything about our products, and I'll provide an answer!")

# Text input for user query
user_input = st.text_input("You:", "")

# If the user has typed something, generate the response
if user_input:
    response = call_rag(user_input)
    st.write(f"Bot: {response}")


def plt_img_base64(img_base64):
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    # Display the image by rendering the HTML
    display(HTML(image_html))


# Example retrieval function to simulate getting documents
def get_relevant_data(query, top_k=3):
    relevant_data = []
    docs = retriever.invoke(query, top_k)
    for doc in docs:
        relevant_data.append(doc.page_content)
        if is_base64(doc.page_content):
            plt_img_base64(doc.page_content)
        else:
            print(doc.page_content)
    return relevant_data

# Streamlit display logic
if user_input:
    relevant_docs = get_relevant_data(user_input)
    system_prompt = f"Answer questions using this data: {relevant_docs}. Be creative with your answers and act like a retail website."   
    response = call_gpt4o(system_prompt, user_input)
    st.write(f"Bot: {response}")

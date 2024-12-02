import tensorflow as tf  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import streamlit as st  # type: ignore
from langchain_community.chat_models import AzureChatOpenAI  # type: ignore
from langchain.schema import SystemMessage, HumanMessage  # type: ignore
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Access the Azure OpenAI API credentials
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
version = os.getenv("AZURE_OPENAI_VERSION")

# Load the trained model
model = load_model('/home/comfortine/Image Classification/Skin_disease_image_classification.keras')  # type: ignore
data_cat = ['acne', 'alopecia areata', 'chickenpox', 'hives', 'melanoma', 'psoriasis', 'ringworm', 'vitiligo', 'warts']
img_height = 180
img_width = 180

# Initialize session state
if "last_disease" not in st.session_state:
    st.session_state.last_disease = None

# Function to get Azure OpenAI response
def get_openai_response(question):
    llm = AzureChatOpenAI(
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        api_version=version,
        model_name="gpt-35-turbo",
        temperature=0.6
    )
    messages = [
        SystemMessage(content="You are a healthcare assistant."),  # type: ignore
        HumanMessage(content=question)  # type: ignore
    ]
    response = llm(messages)
    return response.content

# Streamlit UI
st.header('Self Diagnosis (Skin Disease Analytics Model)')

# File uploader for image input
uploaded_file = st.file_uploader("Upload an Image", type=["jpeg", "jpg", "png"])
default_image = "test3.jpeg"

if uploaded_file is not None:
    # Use uploaded image
    image_path = uploaded_file
    st.image(image_path, width=200, caption='Uploaded Image', use_column_width=True)
else:
    # Use default image
    image_path = default_image
    st.image(image_path, width=200, caption='Default Image (test3.jpeg)', use_column_width=True)

# Load and preprocess the image
try:
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_width, img_height))  # type: ignore
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    # Make predictions
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Update the last predicted disease in session state
    st.session_state.last_disease = data_cat[np.argmax(score)]

    # Display predictions
    st.write('Disease in the image is: ' + st.session_state.last_disease)
    st.write('With an accuracy of {:.2f}%'.format(np.max(score) * 100))
except Exception as e:
    st.error(f"Error processing the image: {e}")

# Chatbot section
st.header("Healthcare Chatbot")

# Input for user questions
input_text = st.text_input("Input: ", key="input")
submit = st.button("Ask Question")

if submit:
    if input_text.strip():
        # Check if the question relates to the last predicted disease
        related_phrases = [
            "recommendations for the disease",
            "health practices for the disease",
            "management of the disease",
            "treatment for the disease"
        ]

        if any(phrase in input_text.lower() for phrase in related_phrases):
            if st.session_state.last_disease:
                input_text = (
                    f"What are the recommendations, health practices, and treatments for "
                    f"{st.session_state.last_disease}?"
                )
            else:
                st.error("No disease has been predicted yet. Please upload an image to make a prediction.")
                input_text = None

        if input_text:  # If input is valid after checks
            response = get_openai_response(input_text)
            st.subheader("The response is")
            st.write(response)
    else:
        st.error("Please enter a question.")






import streamlit as st
import os
import fitz  
from gtts import gTTS
from dotenv import load_dotenv
from summa import summarizer
import math
from langchain_groq import ChatGroq
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize Groq for chat pdf
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.sidebar.error("GROQ_API_KEY is not set. Please set it in the .env file.")
    st.stop()

model = 'llama-3.1-70b-versatile'

groq_chat = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name=model
)

# Initialize Groq for quiz
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)



# Define functions for file handling and processing
def save_uploaded_files(uploaded_files, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for uploaded_file in uploaded_files:
        with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    return save_dir

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def pdf_read(pdf_directory):
    text_content = []
    for file_name in os.listdir(pdf_directory):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(pdf_directory, file_name)
            text = extract_text_from_pdf(file_path)
            text_content.append(text)
    return text_content



def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    return "output.mp3"

def fetch_corrected_text(text_content):

    PROMPT_TEMPLATE = """
    Text: {text_content}
    You are an expert in regenerating provided content in a readable format. 
    Given the above text, regenerate it by making it readable. Make every header text visible to the user.
    Do not change anything, not a single word or information should be missed in your output from the provided content.
    Make sure to keep every word same as it is written in the provided content, you can ignore texts like Copyright or All rights reserved.
    """

    formatted_template = PROMPT_TEMPLATE.format(text_content=text_content)

    # Make API request
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": formatted_template
            }
        ]
    )

# Extract response content
    extracted_response = response.choices[0].message.content
    return extracted_response



def summarize_textrank(text):
    # Generate summary as a list of sentences
    summary = summarizer.summarize(text, split=True)
    return summary

def add_alpha(sentences, summary_sentences, highlight_color="green"):
    def transform(score):
        return math.exp(score * 5)
    
    scores = []
    highlighted_text = ""

    for sentence in sentences:
        if sentence in summary_sentences:
            score = transform(1.0)  # Assign a high score to sentences in the summary
        else:
            score = transform(0.0)  # Assign a low score to non-summary sentences
        
        scores.append(score)

    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score + 1
    
    for i, sentence in enumerate(sentences):
        alpha = round((scores[i] - min_score + 1) / span, 4) * 50
        if alpha > 25:  # Threshold for highlighting
            highlighted_text += f'<span style="background-color:{highlight_color};">{sentence}</span> '
        else:
            highlighted_text += f"{sentence} "
    
    return highlighted_text

# Main app
def main():
    st.set_page_config("Key Sentence Highlighter📝")
    st.header("Key Sentence Highlighter📝")

    # Sidebar
    with st.sidebar:
        st.title("Menu:")
        pdf_files = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        save_dir = "uploaded_pdfs"

    if pdf_files:
        save_uploaded_files(pdf_files, save_dir)
        st.sidebar.success("PDF Uploaded and Processed")


        texts = pdf_read(save_dir)

            # Storing raw text in session_state
        if 'texts' not in st.session_state:
            st.session_state.texts = texts

        text=fetch_corrected_text(st.session_state.texts)

        if text:
                # Split the text into sentences
            sentences = text.split('. ')
    
                # Get summary sentences using TextRank
        summary_sentences = summarize_textrank(text)

            #st.write(text)
    
            # Highlight key sentences
        highlighted_text = add_alpha(sentences, summary_sentences)

            # Display highlighted text
        st.markdown(highlighted_text, unsafe_allow_html=True)
            # Button to convert text to speech
        if st.button("🔊"):
            audio_file = text_to_speech(text)
            st.audio(audio_file, format="audio/mp3")


if __name__ == "__main__":
    main()
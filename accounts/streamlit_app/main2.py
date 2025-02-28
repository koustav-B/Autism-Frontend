import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from my_streamlit_app import (
    record_audio, transcribe_audio, check_grammar, analyze_text_complexity,
    analyze_sentence_structure, analyze_named_entities, analyze_word_variation,
    detect_passive_voice, suggest_simpler_words, custom_grammar_check,
    visualize_pos_tags, visualize_audio, generate_wordcloud
)


# Load Models
emotion_interpreter = tf.lite.Interpreter(model_path="emotion_model.tflite")
emotion_interpreter.allocate_tensors()

sign_interpreter = tf.lite.Interpreter(model_path="sign_language_model.tflite")
sign_interpreter.allocate_tensors()

# Get tensor details for both models
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

sign_input_details = sign_interpreter.get_input_details()
sign_output_details = sign_interpreter.get_output_details()

# Get expected input sizes
EMOTION_IMG_SIZE = emotion_input_details[0]['shape'][1]  # Expected 64
SIGN_IMG_SIZE = sign_input_details[0]['shape'][1]  # Expected 224

# Emotion & Sign Labels
EMOTION_CLASSES = ["Anger", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
SIGN_CLASSES = [chr(i) for i in range(65, 91)]  # A-Z

# PDF Report Generation
def generate_pdf(name, age, email, score, assessment):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Autism Screening Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, f"Name: {name}", ln=True)
    pdf.cell(200, 10, f"Age: {age}", ln=True)
    pdf.cell(200, 10, f"Email: {email}", ln=True)
    pdf.cell(200, 10, f"Score: {score}/40", ln=True)
    pdf.cell(200, 10, f"Assessment: {assessment}", ln=True)
    pdf_file = "autism_report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# Autism Screening Quiz
def autism_screening():
    st.title("🧠 Autism Screening Quiz")
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1, max_value=100, step=1)
    email = st.text_input("Email ID")
    
    questions = [
        "Do you find it difficult to maintain eye contact?",
        "Do you prefer routines and get upset with changes?",
        "Do you struggle to understand jokes or sarcasm?",
        "Do you have difficulty making friends?",
        "Do you get overwhelmed by loud noises or bright lights?",
        "Do you take things literally and struggle with abstract concepts?",
        "Do you feel uncomfortable in social situations?",
        "Do you often repeat certain movements or phrases?",
        "Do you have intense interests in specific topics?",
        "Do you find it difficult to understand others’ emotions?"
    ]
    options = ["Never", "Rarely", "Sometimes", "Often", "Always"]
    points = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
    responses = [points[st.radio(q, options, key=f"q{i}")] for i, q in enumerate(questions)]
    
    if st.button("Submit and Get Report"):
        score = sum(responses)
        assessment = "Likely has autism" if score >= 16 else "Unlikely to have autism"
        pdf_file = generate_pdf(name, age, email, score, assessment)
        
        with open(pdf_file, "rb") as file:
            st.download_button("📥 Download PDF Report", file, "autism_report.pdf", "application/pdf")

# Emotion Detection
def emotion_detection():
    st.title("😊 Autism Emotion Detection")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = img.astype(np.uint8)
        img_resized = cv2.resize(img, (EMOTION_IMG_SIZE, EMOTION_IMG_SIZE))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) / 255.0  # Normalize
        img_final = np.expand_dims(img_gray, axis=[0, -1]).astype(np.float32)

        emotion_interpreter.set_tensor(emotion_input_details[0]['index'], img_final)
        emotion_interpreter.invoke()
        prediction = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

        emotion = EMOTION_CLASSES[np.argmax(prediction)]
        st.image(img, caption=f"Detected Emotion: {emotion}", use_column_width=True)

# Sign Language Recognition
def sign_language_recognition():
    st.title("🤟 Sign Language Recognition (A-Z)")
    uploaded_file = st.file_uploader("Upload an image of a hand sign", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (SIGN_IMG_SIZE, SIGN_IMG_SIZE)) / 255.0
        img_final = np.expand_dims(img_resized, axis=0).astype(np.float32)
        
        sign_interpreter.set_tensor(sign_input_details[0]['index'], img_final)
        sign_interpreter.invoke()
        prediction = sign_interpreter.get_tensor(sign_output_details[0]['index'])
        
        detected_sign = SIGN_CLASSES[np.argmax(prediction)]
        st.image(img, caption=f"Detected Sign: {detected_sign}", use_column_width=True)

# Speech Therapy (Placeholder)

def speech_therapy():
    st.title("🎙 Speech Therapy")

    if st.button("Start Speech Therapy"):
        audio_file = record_audio()
        transcribed_text = transcribe_audio(audio_file)
        
        st.subheader("📝 Transcribed Text")
        st.write(transcribed_text)

        # 📌 Perform Speech Analysis
        check_grammar(transcribed_text)
        analyze_text_complexity(transcribed_text)
        analyze_sentence_structure(transcribed_text)
        analyze_named_entities(transcribed_text)
        analyze_word_variation(transcribed_text)
        detect_passive_voice(transcribed_text)
        suggest_simpler_words(transcribed_text)
        custom_grammar_check(transcribed_text)

        # 📊 Visualizations
        visualize_pos_tags(transcribed_text)
        visualize_audio(audio_file)
        generate_wordcloud(transcribed_text)



# Navigation
def main():
    st.sidebar.title("🔍 Navigation")
    section = st.sidebar.radio("Go to:", ["Autism Screening", "Emotion Detection", "Sign Language", "Speech Therapy"])
    if section == "Autism Screening":
        autism_screening()
    elif section == "Emotion Detection":
        emotion_detection()
    elif section == "Sign Language":
        sign_language_recognition()
    elif section == "Speech Therapy":
        speech_therapy()

# Run App
if __name__ == "__main__":
    main()

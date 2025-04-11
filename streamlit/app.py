import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
import math
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline
)

# --- Configuración general ---
st.set_page_config(page_title="Medication Prediction with BERT", layout="wide")

# --- Estilo visual: centrar contenido y limitar ancho ---
st.markdown("""
    <style>
        .main {
            display: flex;
            justify-content: center;
        }
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
        }
        html, body, [class*="css"] {
            zoom: 1.2;
        }
    </style>
""", unsafe_allow_html=True)

# --- Mostrar estrellas ---
def stars(rating):
    return "⭐" * int(math.floor(rating)) + "☆" * (5 - int(math.floor(rating)))

# --- Cargar modelo principal ---
@st.cache_resource
def load_model_and_tokenizer(model_path, label_encoder_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = joblib.load(label_encoder_path)
    return model, tokenizer, encoder

# --- Traductor ---
@st.cache_resource
def load_translation_pipeline(path):
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return pipeline("translation", model=model, tokenizer=tokenizer)

# --- Resumen ---
@st.cache_resource
def load_summary_pipeline(path):
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

# --- Predicción ---
def predict_medication(text, model, tokenizer, encoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        output = model(**tokens)
        idx = torch.argmax(output.logits, dim=1).item()

    try:
        pred = encoder.inverse_transform([idx])[0]
    except:
        pred = f"(Class {idx} not recognized)"
    return idx, pred

# --- APP ---
def main():
    with st.container():
        st.title("💊 Medication Prediction with BERT")
        st.write("Enter your symptoms and the model will recommend a medication.")

        user_input = st.text_area("📝 Describe your symptoms here...", height=150)

        model_path = "../med_bert_model"
        label_encoder_path = "../med_bert_model/label_encoder.pkl"
        translator_path = "../traductor"
        summarizer_path = "../resumidor"

        model, tokenizer, encoder = load_model_and_tokenizer(model_path, label_encoder_path)
        translator = load_translation_pipeline(translator_path)
        summarizer = load_summary_pipeline(summarizer_path)

        if st.button("🔍 Predict Medication"):
            if user_input.strip():
                with st.spinner("Predicting..."):
                    idx, pred = predict_medication(user_input, model, tokenizer, encoder)
                    st.session_state["predicted_med"] = pred

                    try:
                        df_reviews = pd.read_csv("data/mejores_peores_reviews.csv")
                        df_media = pd.read_csv("data/media_por_medicamento.csv")

                        reviews = df_reviews[df_reviews["drugName_decoded"] == pred]["review"].reset_index(drop=True)
                        st.session_state["reviews"] = reviews

                        if pred in df_media["drugName_decoded"].values:
                            rating = df_media[df_media["drugName_decoded"] == pred]["mean_rating"].values[0]
                            st.session_state["avg_rating"] = rating
                        else:
                            st.session_state["avg_rating"] = None
                    except FileNotFoundError:
                        st.error("❌ CSV files not found.")
                        st.session_state["reviews"] = []
                        st.session_state["avg_rating"] = None
            else:
                st.warning("⚠️ Please enter your symptoms.")

        # Resultados
        if "predicted_med" in st.session_state:
            st.success(f"💡 **Recommended Medication:** {st.session_state['predicted_med']}")
            if st.session_state["avg_rating"]:
                st.markdown(f"⭐ **Average Rating:** {st.session_state['avg_rating']:.1f} {stars(st.session_state['avg_rating'])}")
            else:
                st.info("ℹ️ No average rating available.")

            reviews = st.session_state["reviews"]

            if len(reviews) >= 4:
                st.markdown("### 🟢 Positive Reviews")
                for i in range(2):
                    st.markdown(f"• {reviews[i]}")
                st.markdown("### 🔴 Negative Reviews")
                for i in range(2, 4):
                    st.markdown(f"• {reviews[i]}")

                if st.button("🌐 Translate Reviews"):
                    with st.spinner("Translating..."):
                        translated = translator(list(reviews[:4]))
                        st.markdown("### 🌐 Translated Reviews")
                        for t in translated:
                            st.markdown(f"• {t['translation_text']}")

                if st.button("📋 Summarize Reviews"):
                    with st.spinner("Summarizing..."):
                        summary = summarizer(" ".join(reviews[:4]), max_length=100, min_length=30)[0]['summary_text']
                        st.markdown("### 📋 Review Summary")
                        st.write(summary)
            else:
                st.info("ℹ️ Not enough reviews available.")

if __name__ == "__main__":
    main()

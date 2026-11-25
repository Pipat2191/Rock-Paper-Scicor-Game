import random

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Rock Paper Scissors AI", page_icon="✊")

MODEL_PATH = "keras_model.h5"   # exported from Teachable Machine
LABELS_PATH = "labels.txt"      # exported from Teachable Machine
BEATS = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
EMOJI = {"Rock": "✊", "Paper": "✋", "Scissors": "✌️"}


@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH) as f:
        # lines look like "0 Rock" -> keep "Rock"
        labels = [line.strip().split(" ", 1)[1] for line in f if line.strip()]
    return model, labels


def predict(image, model, labels):
    image = ImageOps.fit(image.convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    arr = (np.asarray(image).astype(np.float32) / 127.5) - 1  # TM normalisation
    probs = model.predict(arr[np.newaxis, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])


def winner(player, computer):
    if player == computer:
        return "Draw"
    return "You win!" if BEATS[player] == computer else "Computer wins!"


if "score" not in st.session_state:
    st.session_state.score = {"You": 0, "Computer": 0, "Draws": 0}

st.title("✊✋✌️ Rock Paper Scissors AI")
st.write("Show your hand to the camera and the AI will recognise it, then play against the computer.")

model, labels = load_assets()
photo = st.camera_input("Take a photo of your hand")

if photo:
    player, confidence = predict(Image.open(photo), model, labels)
    computer = random.choice(list(BEATS))
    result = winner(player, computer)

    st.subheader(f"AI sees: {EMOJI.get(player, '')} {player} ({confidence:.0%} sure)")
    st.subheader(f"Computer plays: {EMOJI[computer]} {computer}")
    st.header(result)

    if confidence < 0.6:
        st.warning("The AI is not very sure. Try better light and a plain background.")

    if st.button("Count this round"):
        key = {"You win!": "You", "Computer wins!": "Computer"}.get(result, "Draws")
        st.session_state.score[key] += 1

s = st.session_state.score
st.divider()
st.write(f"**Score** — You: {s['You']} | Computer: {s['Computer']} | Draws: {s['Draws']}")

st.caption(
    "⚠️ For learning only. AI can be wrong: it may misread your hand, especially in "
    "poor light or with a busy background. It only knows rock, paper and scissors."
)

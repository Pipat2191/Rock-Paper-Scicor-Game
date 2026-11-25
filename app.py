import os
import random
from collections import Counter

import streamlit as st

st.set_page_config(page_title="Rock Paper Scissors AI", page_icon="✊", layout="centered")

MOVES = ["Rock", "Paper", "Scissors"]
EMOJI = {"Rock": "✊", "Paper": "✋", "Scissors": "✌️"}
BEATS = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}   # key beats value
COUNTER = {v: k for k, v in BEATS.items()}                           # move that beats key
MODEL_PATH, LABELS_PATH = "keras_model.h5", "labels.txt"


# ---------- game logic ----------
def new_game():
    st.session_state.score = {"You": 0, "Computer": 0, "Draws": 0}
    st.session_state.history = []      # list of (player, computer, result)
    st.session_state.last = None


def decide(player, computer):
    if player == computer:
        return "Draw"
    return "You" if BEATS[player] == computer else "Computer"


def computer_move(level):
    """Easy = random. Smart = learns your habits and plays what beats your likely move."""
    moves = [h[0] for h in st.session_state.history]
    if level == "Easy (random)" or len(moves) < 3:
        return random.choice(MOVES)
    # Look at what you played after your last move, in the past
    follow = Counter(moves[i + 1] for i in range(len(moves) - 1) if moves[i] == moves[-1])
    pool = follow if follow else Counter(moves)
    predicted = pool.most_common(1)[0][0]
    return COUNTER[predicted]


def play_round(player, level):
    comp = computer_move(level)
    result = decide(player, comp)
    key = result if result in ("You", "Computer") else "Draws"
    st.session_state.score[key] += 1
    st.session_state.history.append((player, comp, result))
    st.session_state.last = (player, comp, result)


# ---------- camera (optional AI model) ----------
@st.cache_resource
def load_model_and_labels():
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH) as f:
        labels = [l.strip().split(" ", 1)[1] for l in f if l.strip()]
    return model, labels


def predict_hand(photo):
    import numpy as np
    from PIL import Image, ImageOps
    model, labels = load_model_and_labels()
    img = ImageOps.fit(Image.open(photo).convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    arr = (np.asarray(img).astype("float32") / 127.5) - 1
    probs = model.predict(arr[None, ...], verbose=0)[0]
    i = int(np.argmax(probs))
    return labels[i], float(probs[i])


# ---------- UI ----------
if "score" not in st.session_state:
    new_game()

st.title("✊✋✌️ Rock Paper Scissors AI")

with st.sidebar:
    st.header("Settings")
    level = st.radio("Computer level", ["Easy (random)", "Smart (learns your habits)"])
    target = st.selectbox("First to win…", [3, 5, 10], index=0)
    if st.button("🔄 New game"):
        new_game()
        st.rerun()

score = st.session_state.score
c1, c2, c3 = st.columns(3)
c1.metric("You", score["You"])
c2.metric("Computer", score["Computer"])
c3.metric("Draws", score["Draws"])

game_over = max(score["You"], score["Computer"]) >= target

tab_buttons, tab_camera = st.tabs(["🖱️ Play with buttons", "📷 Play with camera (AI)"])

with tab_buttons:
    if not game_over:
        cols = st.columns(3)
        for col, m in zip(cols, MOVES):
            if col.button(f"{EMOJI[m]} {m}", use_container_width=True, key=f"btn_{m}"):
                play_round(m, level)
                st.rerun()
    else:
        if score["You"] > score["Computer"]:
            st.success("🏆 You won the game!")
            st.balloons()
        else:
            st.error("🤖 The computer won the game. Try again!")
        st.write("Press **New game** in the sidebar to play again.")

with tab_camera:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        st.info(
            "Camera mode needs a Teachable Machine model. Add `keras_model.h5` and "
            "`labels.txt` to the repo (see README). Button mode works without it."
        )
    elif game_over:
        st.write("Game over. Press **New game** in the sidebar.")
    else:
        photo = st.camera_input("Show rock, paper or scissors")
        if photo:
            try:
                move, conf = predict_hand(photo)
            except Exception as e:  # missing tensorflow, bad model, etc.
                st.error(f"Could not run the model: {e}")
            else:
                st.write(f"AI sees: **{EMOJI.get(move, '')} {move}** ({conf:.0%} sure)")
                if conf < 0.6:
                    st.warning("Not very sure. Try better light and a plain background.")
                if move in MOVES and st.button("✅ Use this move"):
                    play_round(move, level)
                    st.rerun()

# ---------- result of last round ----------
st.divider()
if st.session_state.last:
    p, c, r = st.session_state.last
    st.subheader(f"You {EMOJI[p]}  vs  {EMOJI[c]} Computer")
    if r == "Draw":
        st.info("It's a draw!")
    elif r == "You":
        st.success(f"You win this round! {p} beats {c}.")
    else:
        st.error(f"Computer wins this round. {c} beats {p}.")

if st.session_state.history:
    with st.expander("Round history"):
        for i, (p, c, r) in enumerate(reversed(st.session_state.history), 1):
            n = len(st.session_state.history) - i + 1
            st.write(f"Round {n}: {EMOJI[p]} {p} vs {EMOJI[c]} {c} → {r if r == 'Draw' else r + ' won'}")

st.caption(
    "⚠️ For learning only. AI can be wrong: the camera model may misread your hand, and the "
    "'Smart' computer only guesses from your past moves. It is not always right."
)

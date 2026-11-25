# Rock Paper Scissors AI ✊✋✌️

A complete rock paper scissors game built with Streamlit.

## Who is it for?
Anyone who wants a fun way to see how AI can learn patterns. Made for an AI Literacy class project.

## Features
- Play with buttons: Rock, Paper or Scissors
- Score, round history and "first to 3 / 5 / 10" game mode
- Two computer levels: **Easy** (random) and **Smart** (learns which move you tend to play next and counters it)
- Optional **camera mode**: a Teachable Machine model recognises your hand

## Files
- `app.py` - the app
- `requirements.txt` - packages
- `keras_model.h5`, `labels.txt` - *optional* Teachable Machine model for camera mode

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Enable camera mode
1. Train a Teachable Machine image model with classes `Rock`, `Paper`, `Scissors`.
2. Export as TensorFlow > Keras, and put `keras_model.h5` and `labels.txt` next to `app.py`.
3. In `requirements.txt`, uncomment `tensorflow-cpu==2.15.0`.
4. On Streamlit Cloud, choose Python 3.11 in Advanced settings.

## Limits (please read)
- For learning only. AI can be wrong.
- The Smart computer only guesses from your past moves, so it can lose, especially if you play randomly.
- The camera model was trained on a small set of photos, so it may fail with different light, backgrounds or hands.

## Team
Add member names here.

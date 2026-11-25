# Rock Paper Scissors AI ✊✋✌️

A Streamlit app that uses a camera and an image model to recognise your hand
(rock, paper or scissors) and plays a round against the computer.

## Who is it for?
Anyone who wants a fun way to see how image AI works. Made for an AI Literacy class project.

## How it works
1. The model was trained with **Teachable Machine** using webcam photos of 3 hand shapes.
2. `app.py` loads the exported model (`keras_model.h5`, `labels.txt`).
3. The AI predicts your move, the computer picks randomly, and the winner is shown.

## Files
- `app.py` – the Streamlit app
- `requirements.txt` – Python packages
- `keras_model.h5`, `labels.txt` – the trained model

## Limits (please read)
- AI can be wrong. It was trained on a small set of photos, so it may fail with
  different lighting, backgrounds, skin tones or hand angles.
- It only knows 3 classes. Anything else will still be guessed as one of them.
- Made for learning only.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```



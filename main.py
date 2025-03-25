import os
import numpy as np
import cv2
import spotipy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from spotipy.oauth2 import SpotifyClientCredentials
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Paths to dataset directories
MODEL_PATH = "emotion_model.h5"

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")


def load_emotion_model():
    """Load the trained emotion model."""
    if not os.path.exists(MODEL_PATH):
        print("Trained model not found! Exiting...")
        exit()
    return load_model(MODEL_PATH)


def get_playlist_url(emotion):
    """Return a Spotify playlist URL based on the detected emotion."""
    emotion_to_playlist = {
        "Happy": "spotify:playlist:37i9dQZF1DXdPec7aLTmlC",
        "Sad": "spotify:playlist:37i9dQZF1DX7qK8ma5wgG1",
        "Angry": "spotify:playlist:37i9dQZF1DX6GJXiuZRisr",
        "Neutral": "spotify:playlist:37i9dQZF1DX4WYpdgoIcn6",
        "Fear": "spotify:playlist:37i9dQZF1DWSqBruwoIXkA",
        "Surprise": "spotify:playlist:37i9dQZF1DX4sWSpwq3LiO",
        "Disgust": "spotify:playlist:37i9dQZF1DWTLrNDPW5co3"
    }
    playlist_uri = emotion_to_playlist.get(emotion, "spotify:playlist:37i9dQZF1DX4WYpdgoIcn6")
    return f"https://open.spotify.com/playlist/{playlist_uri.split(':')[-1]}"


def detect_emotion(frame, model, face_cascade):
    """Detect emotions from a given frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        predictions = model.predict(roi_gray)
        return EMOTION_LABELS[np.argmax(predictions)]

    return None


def detect_emotion_live():
    """Detect emotions in real-time using webcam for 3 seconds and suggest a playlist."""
    model = load_emotion_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    emotions_detected = []
    start_time = time.time()

    while time.time() - start_time < 3:  # Run for 3 seconds
        ret, frame = cap.read()
        if not ret:
            break

        detected_emotion = detect_emotion(frame, model, face_cascade)
        if detected_emotion:
            emotions_detected.append(detected_emotion)

    cap.release()
    cv2.destroyAllWindows()

    if emotions_detected:
        most_frequent_emotion = max(set(emotions_detected), key=emotions_detected.count)
        print(f"Detected emotion: {most_frequent_emotion}")
        print(f"Open this playlist: {get_playlist_url(most_frequent_emotion)}")
    else:
        print("No face detected. Please try again.")


if __name__ == "__main__":
    detect_emotion_live()
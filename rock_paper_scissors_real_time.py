"""
Uses the RPC model to make predictions from a video feed.
"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_frame(frame, model):
    """
    Preprocesses the frame and uses the model to predict the gesture.
    """
    img = cv2.resize(frame, (150, 150))  # Resize to match the input shape of the model
    img = img.astype('float32') / 255  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return np.argmax(prediction)

def run_rock_paper_scissors(model_path):
    """
    Captures video from the webcam and uses the model to predict gestures in real-time.
    """
    model = load_model(model_path)  # Load the pre-trained model
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Couldn't capture a frame.")
            break

        prediction = predict_frame(frame, model)
        gesture = ['Paper', 'Rock', 'Scissors'][prediction]
        
        cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Rock Paper Scissors', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
run_rock_paper_scissors('./models/RPSClassifier.h5')

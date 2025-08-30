# Doodle Classifier Web App

A web application for classifying hand-drawn doodles using a PyTorch CNN model.

## Setup

### Backend
1. Navigate to `backend/`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Ensure `classes.txt` contains the class names (one per line).
4. Place the trained model (`doodle_model.pth`) in `backend/models/`.
5. Run the Flask server: `python app.py`.

### Frontend
1. Navigate to `frontend/`.
2. Open `index.html` in a web browser (no build step required).

### Training
1. Ensure the QuickDraw dataset `.npy` files are in `backend/dataset/`.
2. Run `python train.py` to train and save the model.

## Usage
- Draw a doodle on the canvas.
- Click "Predict" to classify the doodle.
- Click "Clear" to reset the canvas.

## Notes
- The model expects 32x32 grayscale images with padding.
- The backend runs on `http://localhost:5000`.
- The frontend is a static HTML/CSS/JS app.
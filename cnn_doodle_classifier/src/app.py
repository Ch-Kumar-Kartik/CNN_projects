import torch #type:ignore
# from cnn_doodle_classifier.src.model import DoodleCNN
from model import DoodleCNN
from flask import Flask, render_template, request, jsonify #type:ignore
import numpy as np #type:ignore
from PIL import Image
import io
import base64
import torch #type:ignore
import torch.nn as nn #type:ignore
import torch.optim as optim #type:ignore
import torchvision #type:ignore
import torchvision.transforms as transforms #type:ignore
from torch.utils.data import DataLoader, Dataset #type:ignore
import numpy as np #type:ignore

# app = Flask(__name__)

# # Load the pre-trained model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DoodleCNN(num_classes=10).to(device)
# model.load_state_dict(torch.load("/data/Projects/cnn_projects/cnn_doodle_classifier/saved_models/doodle_cnn.pth"))
# model.eval()
# print("Model loaded for web app")

# # Define category labels
# categories = ["cat", "dog", "bird", "fish", "tree", "flower", "car", "house", "sun", "moon"]

# # Transform for preprocessing (match training transform)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# @app.route('/')
# def index():
#     return render_template('/data/Projects/cnn_projects/cnn_doodle_classifier/src/index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the base64-encoded image data from the request
#     data = request.json['image']
#     # Remove the data URL prefix (e.g., "data:image/png;base64,")
#     header, encoded = data.split(',')
#     image_data = base64.b64decode(encoded)
#     image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
#     image = image.resize((28, 28), Image.Resampling.LANCZOS)  # Resize to 28x28
#     image = np.array(image) / 255.0  # Normalize to [0, 1]
#     image = image.reshape(1, 28, 28)  # Add batch dimension
#     image = transform(image).to(device)  # Apply the same transform as training
    
#     # Predict
#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output.data, 1)
#         prediction = categories[predicted.item()]
#         confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()
    
#     return jsonify({'prediction': prediction, 'confidence': confidence})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

app = Flask(__name__, template_folder='/data/Projects/cnn_projects/cnn_doodle_classifier/templates')

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DoodleCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("/data/Projects/cnn_projects/cnn_doodle_classifier/saved_models/doodle_cnn.pth"))
model.eval()
print("Model loaded for web app")

# Define category labels
categories = ["cat", "dog", "bird", "fish", "tree", "flower", "car", "house", "sun", "moon"]

# Transform for preprocessing (match training transform)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    header, encoded = data.split(',')
    image_data = base64.b64decode(encoded)

    # Preprocess
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = transform(image).unsqueeze(0).to(device).float()

    # Predict
    with torch.no_grad():
        output = model(image)
        print(f"Raw output: {output}")  # Debug print
        _, predicted = torch.max(output.data, 1)
        prediction = categories[predicted.item()]
        confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()
    
    return jsonify({'prediction': prediction, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
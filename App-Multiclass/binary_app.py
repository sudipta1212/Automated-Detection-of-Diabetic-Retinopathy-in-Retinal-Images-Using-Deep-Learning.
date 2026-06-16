import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import bcrypt
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'ML' 

# ----------------------------------- MongoDB setup --------------------------------------

uri = "mongodb+srv://sudipta1212das32_db_user:JyaWunlzZwtIXNH2@cluster0.a1nme1b.mongodb.net/?appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"MongoDB connection error: {e}")

# Create database and collection
db = client['Diabetic_Retinopathy']
collection = db['binary_prediction_logs']
user_collection = db['user_logs']

# ----------------------------------- Model Setup --------------------------------------
# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of classes
NUM_CLASSES = 1

# Define image transform
transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class names
class_names = [
    'Diabetic Retina',
    'Non Diabetic Retina'
]

# Load model architecture
def load_mobilenet_v3_large(num_classes):
    model = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT')
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, num_classes)
    return model

# Define model path
MODEL_PATH = 'mobilenetv3largemodel_epoch_9.pth'

model = load_mobilenet_v3_large(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----------------------------------- API setup --------------------------------------
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    try:
        # Load image and preprocess
        image = Image.open(image_file).convert('RGB')
        image = transform(image).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(image).squeeze()
            prob = torch.sigmoid(outputs).item()

        # Prediction
        predicted_idx = 1 if prob >= 0.5 else 0
        predicted_label = class_names[predicted_idx]
        confidence_score = prob if predicted_idx == 1 else 1 - prob

        # Optional MongoDB insert (comment if not needed)
        mongo_data = {
            'image': image_file.filename,
            'predicted_class': predicted_idx,
            'predicted_label': predicted_label,
            'confidence': confidence_score
        }
        try:
            collection.insert_one(mongo_data)
            print(f"Inserted document into MongoDB: {mongo_data}")
        except Exception as e:
            print(f"Error inserting into MongoDB: {e}")

        return jsonify({
            'predicted_label': predicted_label,
            'confidence': confidence_score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Routes for signup and login
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        if user_collection.find_one({'username': username}):
            return jsonify({'error': 'Username already exists'}), 409

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_collection.insert_one({'username': username, 'password': hashed_password})
        return jsonify({'message': 'User registered successfully'}), 201

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')

        user = user_collection.find_one({'username': username})
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return jsonify({'error': 'Invalid username or password'}), 401

        session['username'] = username
        return jsonify({'message': 'Login successful'}), 200

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

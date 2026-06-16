import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import bcrypt
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

# ----------------------------------- Flask setup --------------------------------------
app = Flask(__name__)
app.secret_key = 'ML'

# ----------------------------------- MongoDB setup --------------------------------------
#username: sudipta1212das32_db_user
#password: JyaWunlzZwtIXNH2
uri = "mongodb+srv://sudipta1212das32_db_user:JyaWunlzZwtIXNH2@cluster0.a1nme1b.mongodb.net/?appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"MongoDB connection error: {e}")

db              = client['Diabetic_Retinopathy']
collection      = db['prediction_logs']
user_collection = db['user_logs']

# ----------------------------------- Model Setup --------------------------------------
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

CLASS_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Mild non-proliferative diabetic retinopathy.",
    2: "Moderate non-proliferative diabetic retinopathy.",
    3: "Severe non-proliferative diabetic retinopathy.",
    4: "Proliferative diabetic retinopathy — advanced stage.",
}

# Transform — identical to val_transform in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

MODEL_PATH = r"best_model.pth"

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"[✓] EfficientNet-B0 loaded from: {MODEL_PATH}")

# LIME explainer — initialised once at startup (lightweight)
explainer = lime_image.LimeImageExplainer(random_state=42)
print("[✓] LIME explainer ready.")

# ----------------------------------- LIME helpers ------------------------------------
def make_predict_fn():
    """
    Returns a LIME-compatible predict function.
    LIME passes batches of uint8 numpy arrays (H, W, 3) → model returns (N, 5) probs.
    """
    def predict_fn(images: np.ndarray) -> np.ndarray:
        batch = []
        for img_arr in images:
            pil = Image.fromarray(img_arr.astype(np.uint8), mode="RGB")
            batch.append(transform(pil))
        tensor = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()
        return probs
    return predict_fn

predict_fn = make_predict_fn()


def run_lime(pil_img: Image.Image, pred_class: int,
             num_samples: int = 2000, num_features: int = 10):
    """Run LIME and return the side-by-side figure as base64 PNG."""
    img_arr = np.array(pil_img.resize((224, 224))).astype(np.uint8)

    explanation = explainer.explain_instance(
        img_arr,
        predict_fn,
        top_labels=NUM_CLASSES,
        hide_color=None,
        num_samples=num_samples,
        batch_size=32,
        random_seed=42,
    )

    temp, mask = explanation.get_image_and_mask(
        pred_class,
        positive_only=False,
        num_features=num_features,
        hide_rest=False,
    )

    lime_overlay = mark_boundaries(temp.astype(np.float64) / 255.0, mask)

    # Side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e2e")

    axes[0].imshow(img_arr)
    axes[0].set_title("Original Image",   color="white", fontsize=12, pad=8)
    axes[0].axis("off")

    axes[1].imshow(lime_overlay)
    axes[1].set_title("LIME Explanation", color="white", fontsize=12, pad=8)
    axes[1].axis("off")

    green_patch = mpatches.Patch(color="lime", label="Supports prediction")
    red_patch   = mpatches.Patch(color="red",  label="Contradicts prediction")
    axes[1].legend(handles=[green_patch, red_patch],
                   loc="lower right", fontsize=8,
                   framealpha=0.6, facecolor="#1e1e2e", labelcolor="white")

    fig.suptitle(
        f"Predicted: {CLASS_NAMES[pred_class]}",
        fontsize=13, fontweight="bold", color="white", y=1.01,
    )

    plt.tight_layout()

    # Encode to base64 — no file saving needed
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ----------------------------------- Routes ------------------------------------------
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/explain_page')
def explain_page():
    return render_template('explain.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    try:
        pil_img = Image.open(image_file).convert('RGB')
        tensor  = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu()[0].tolist()

        predicted_idx   = int(probs.index(max(probs)))
        predicted_label = CLASS_NAMES[predicted_idx]
        confidence      = round(probs[predicted_idx] * 100, 2)
        description     = CLASS_DESCRIPTIONS[predicted_idx]
        all_probs       = {CLASS_NAMES[i]: round(probs[i] * 100, 2)
                           for i in range(NUM_CLASSES)}

        # MongoDB log
        try:
            collection.insert_one({
                'image'           : image_file.filename,
                'predicted_class' : predicted_idx,
                'predicted_label' : predicted_label,
                'confidence'      : confidence,
                'all_probs'       : all_probs,
            })
        except Exception as e:
            print(f"MongoDB insert error: {e}")

        return jsonify({
            'predicted_label' : predicted_label,
            'predicted_class' : predicted_idx,
            'confidence'      : confidence,
            'description'     : description,
            'all_probs'       : all_probs,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain', methods=['POST'])
def explain():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    try:
        pil_img = Image.open(image_file).convert('RGB')
        tensor  = transform(pil_img).unsqueeze(0).to(DEVICE)

        # Get predicted class for LIME target
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu()[0].tolist()

        pred_class = int(probs.index(max(probs)))

        # Run LIME — returns base64 PNG string
        img_b64 = run_lime(pil_img, pred_class)

        return jsonify({
            'lime_image'      : f"data:image/png;base64,{img_b64}",
            'predicted_label' : CLASS_NAMES[pred_class],
            'confidence'      : round(probs[pred_class] * 100, 2),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------- Auth routes -------------------------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data     = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        if user_collection.find_one({'username': username}):
            return jsonify({'error': 'Username already exists'}), 409

        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_collection.insert_one({'username': username, 'password': hashed})
        return jsonify({'message': 'User registered successfully'}), 201

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data     = request.json
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
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import io
from model import UNetColorization, colorize_image  # Assuming you save the model code in model.py

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'unet_colorization_veg_valid.pth'  # Update with your model path

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetColorization().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'colorized_{filename}')
        file.save(input_path)
        
        try:
            # Colorize the image
            colorized_path, comparison_path = colorize_image(model, input_path, output_path, device)
            
            # Read the comparison image and convert to base64
            with open(comparison_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Clean up temporary files
            os.remove(input_path)
            os.remove(colorized_path)
            os.remove(comparison_path)
            
            return jsonify({
                'status': 'success',
                'image': img_data
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
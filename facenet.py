import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Path to the directory containing your images
images_dir = 'dataset'

# Path to the target image
target_image_path = '9.jpg'

# Extract features from the target image
target_features = extract_features(target_image_path, model)

# Initialize variables to store best match
best_match = None
best_similarity = -1

# Iterate over each image in the directory
for filename in os.listdir(images_dir):
    if filename.endswith('.jpeg'):
        image_path = os.path.join(images_dir, filename)
        # Extract features from the current image
        features = extract_features(image_path, model)
        # Calculate cosine similarity between the target image and the current image
        cosine_similarity = np.dot(target_features, features) / (np.linalg.norm(target_features) * np.linalg.norm(features))
        # Update best match if similarity is higher
        if cosine_similarity > best_similarity:
            best_similarity = cosine_similarity
            best_match = filename

# Set a threshold for similarity
similarity_threshold = 0.9
print(best_similarity,similarity_threshold)
# Output the best match
if best_similarity > 0:
    print(f"Best match: {best_match} (Similarity Score = {best_similarity})")
else:
    print("No perfect match found.")

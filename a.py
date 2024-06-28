import face_recognition
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def encode_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings, face_locations, image

def compare_faces(input_image_encodings, input_image_locations, input_image, folder_path):
    files = os.listdir(folder_path)
    match_found = False

    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file)
            folder_image_encodings, folder_image_locations, folder_image = encode_faces(image_path)
            for input_encoding, input_location in zip(input_image_encodings, input_image_locations):
                for folder_encoding, folder_location in zip(folder_image_encodings, folder_image_locations):
                    face_distance = face_recognition.face_distance([input_encoding], folder_encoding)
                    if face_distance < 0.6:
                        match_found = True

                        # Display the matched images
                        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                        ax[0].imshow(input_image[input_location[0]:input_location[2], input_location[3]:input_location[1]])
                        ax[0].set_title("Input Face")
                        ax[0].axis('off')
                        folder_image = mpimg.imread(image_path)
                        ax[1].imshow(folder_image[folder_location[0]:folder_location[2], folder_location[3]:folder_location[1]])
                        ax[1].set_title(f"Matched Face: {file}")
                        ax[1].axis('off')
                        plt.show()
                        break  # Exit inner loop since match is found for this input face
                if match_found:
                    break  # Exit outer loop since match is found for this input face
            if match_found:
                break  # Exit loop since match is found for any face in the input image

    if not match_found:
        print("No match found.")

input_image_path = 'manojsir1.jpeg'
folder_path = "dataset"

print(f"Comparing faces in '{input_image_path}' with images in the folder...")
input_image_encodings, input_image_locations, input_image = encode_faces(input_image_path)
compare_faces(input_image_encodings, input_image_locations, input_image, folder_path)

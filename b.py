# import face_recognition
# import os

# def compare_faces(input_image_path, folder_path):
#     input_image = face_recognition.load_image_file(input_image_path)
#     input_encodings = face_recognition.face_encodings(input_image)
    
#     if len(input_encodings) == 0:
#         print("No faces detected in the input image.")
#         return

#     input_encoding = input_encodings[0]

#     files = os.listdir(folder_path)
#     match_found = False

#     for file in files:
#         if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(folder_path, file)
#             folder_image = face_recognition.load_image_file(image_path)
#             folder_encodings = face_recognition.face_encodings(folder_image)
            
#             if len(folder_encodings) == 0:
#                 continue
            
#             folder_encoding = folder_encodings[0]

#             face_distance = face_recognition.face_distance([input_encoding], folder_encoding)
#             if face_distance < 0.6:
#                 match_found = True
#                 print(f"Match found: {file}")
#                 break

#     if not match_found:
#         print("No match found.")

# input_image_path = 'notmanoj.jpeg'
# folder_path = "dataset"

# print(f"Comparing faces in '{input_image_path}' with images in the folder...")
# compare_faces(input_image_path, folder_path)

import os
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_dataset(dataset_folder):
    dataset = {}
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(dataset_folder, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            dataset[name] = (image, face_encoding)
    return dataset

def find_face_match(image_to_search, dataset):
    # Find face encodings for the image to search
    face_encodings = face_recognition.face_encodings(image_to_search)

    matches = []
    for face_encoding in face_encodings:
        # Compare face encodings with the dataset
        matches = face_recognition.compare_faces([x[1] for x in dataset.values()], face_encoding)
        if True in matches:
            matched_index = matches.index(True)
            matched_name = list(dataset.keys())[matched_index]
            matches.append((matched_name, dataset[matched_name][0]))

    return matches

# Example usage
if __name__ == "__main__":
    # Load the dataset
    dataset_folder = "dataset"
    dataset = load_dataset(dataset_folder)

    # Load the image to search
    image_to_search = face_recognition.load_image_file('manojsir2.jpeg')

    # Find the match
    matches = find_face_match(image_to_search, dataset)

    if matches:
        print("Face(s) found!")
        for match in matches:
            print("Matched Name:", match[0])

        # Display both images
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, len(matches)+1)

        ax = plt.subplot(gs[0])
        ax.imshow(image_to_search)
        ax.set_title('Image to Search')
        ax.axis('off')

        for i, match in enumerate(matches):
            ax = plt.subplot(gs[i+1])
            ax.imshow(match[1])
            ax.set_title(f'Matched Image: {match[0]}')
            ax.axis('off')

        plt.show()
    else:
        print("Face not found.")

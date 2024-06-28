# import face_recognition

# def compare_faces(image1_path, image2_path):
#     # Load the images
#     image1 = face_recognition.load_image_file(image1_path)
#     image2 = face_recognition.load_image_file(image2_path)
    
#     image1_encodings = face_recognition.face_encodings(image1)
#     image2_encodings = face_recognition.face_encodings(image2)
    
#     if len(image1_encodings) == 0 or len(image2_encodings) == 0:
#         return False  # No faces found in one of the images
    
#     image1_encoding = image1_encodings[0]
#     image2_encoding = image2_encodings[0]
    
#     # Compare faces
#     results = face_recognition.compare_faces([image1_encoding], image2_encoding)
    
#     return results[0]

# if __name__ == "__main__":
#     image1_path = "images/KatrinaKaif1.jpg"  # Path to the first image
#     image2_path = "images/13glams2.jpg"  # Path to the second image
    
#     # Compare faces
#     faces_match = compare_faces(image1_path, image2_path)
    
#     if faces_match:
#         print("The faces match!")
#     else:
#         print("The faces do not match.")



import face_recognition

def find_face_match(image_to_search, library_images):
    # Find face encodings for the image to search
    image_to_search_encoding = face_recognition.face_encodings(image_to_search)[0]

    for library_image in library_images:
        # Find face encodings for the library image
        library_image_encoding = face_recognition.face_encodings(library_image)[0]

        # Compare face encodings
        match = face_recognition.compare_faces([library_image_encoding], image_to_search_encoding)

        if match[0]:
            return True, library_image

    # If no match is found
    return False, None

# Example usage
if __name__ == "__main__":
    # Load the image to search
    image_to_search = face_recognition.load_image_file('manojsir1.jpeg')

    # Load the library images
    library_images = []
    library_images.append(face_recognition.load_image_file('manojsir2.jpeg'))
    # library_images.append(face_recognition.load_image_file('102.jpeg'))
    # Add more images as needed

    # Find the match
    found, matched_image = find_face_match(image_to_search, library_images)

    if found:
        print("Face found!")
        # Do something with matched_image
    else:
        print("Face not found.")

import face_recognition
import os
from shutil import copyfile
from joblib import Parallel, delayed

# Author: Guilherme Dall'Agnol Deconto
# Version: 1.0
# Date: 02/12/2021

# Array of images to learn
images = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg", "image6.jpg", "image7.jpg", "image8.jpg", "image9.jpg", "image10.jpg"]

known_face_encodings = []

# Returns current working directory
directory = os.getcwd()

# Load image and learn how to recognize it
for image in images:
    image_file = face_recognition.load_image_file(image)
    face_encoding = face_recognition.face_encodings(image_file)[0]
    known_face_encodings.append(face_encoding)

def find(filename):
    # Check the file extension to see if it's an image (jpg, png, etc)
    if filename.endswith(".jpg"):
        # Print filename
         print(filename)
         # Load the image into a Python variable
         new_picture = face_recognition.load_image_file(filename)
         # Loop through each face in the image
         for face_encoding in face_recognition.face_encodings(new_picture):
            # See if the face is a match for the known face(s)
            # Tolerance is the distance between faces
            # In this case I'm using 0.5 which is more strict
            # But you can use a higher value if you want more performance and more matches
             results = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)
             # If it's a match, save the image to another folder
             if any(results) == True:
                 copyfile(filename, "./foundthem/" + filename)
                 continue
                 
# Loop through each image in the directory
# Parallelize the loop to speed up the process, but you can use a single loop if you want
# This code will run in parallel with all available cores
# If you want to use a specific number of cores, for example 4 cores, you can use the following code:
# Parallel(n_jobs=4)(delayed(find)(filename) for filename in os.listdir(directory))
Parallel(n_jobs=-1)(delayed(find)(image) for image in os.listdir(directory))
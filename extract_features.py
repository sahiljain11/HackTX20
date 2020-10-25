import io
import os
from google.cloud import vision
from google.cloud.vision import types
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.spatial.distance as sp
import scipy.io.wavfile as wav
from python_speech_features import logfbank
from python_speech_features import mfcc
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import itertools
from mpl_toolkits import mplot3d


# Calls the google api to gather the facial landmarks
def call_google_vision (path_to_file):
    # Instantiate the client
    client = vision.ImageAnnotatorClient()
    # Grab the file you want annotate
    file_name = os.path.abspath(path_to_file)
    # Load the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    # Performs label detection on the image file
    response = client.face_detection(image=image)
    faces = response.face_annotations
    # Names of likelihood from google.cloud.vision.enums
    #likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE','LIKELY', 'VERY_LIKELY')
    #landmark_names = ("LEFT_EYE", "RIGHT_EYE", "LEFT_OF_LEFT_EYEBROW", "RIGHT_OF_LEFT_EYEBROW", "LEFT_OF_RIGHT_EYEBROW", "RIGHT_OF_RIGHT_EYEBROW", "MIDPOINT_BETWEEN_EYES", "NOSE_TIP", "UPPER_LIP", "LOWER_LIP", "MOUTH_LEFT", "MOUTH_RIGHT", "MOUTH_CENTER", "NOSE_BOTTOM_RIGHT", "NOSE_BOTTOM_LEFT", "NOSE_BOTTOM_CENTER", "LEFT_EYE_TOP_BOUNDARY", "LEFT_EYE_RIGHT_CORNER", "LEFT_EYE_BOTTOM_BOUNDARY", "LEFT_EYE_LEFT_CORNER", "RIGHT_EYE_TOP_BOUNDARY", "RIGHT_EYE_RIGHT_CORNER", "RIGHT_EYE_BOTTOM_BOUNDARY", "RIGHT_EYE_LEFT_CORNER", "LEFT_EYEBROW_UPPER_MIDPOINT", " RIGHT_EYEBROW_UPPER_MIDPOINT", "LEFT_EAR_TRAGION", "RIGHT_EAR_TRAGION", "FOREHEAD_GLABELLA", "CHIN_GNATHION", "CHIN_LEFT_GONION", "CHIN_RIGHT_GONION", "35", "36")
    return faces, response 

# Extracts the landmark from the faces
def extract_landmarks (faces, response):
    # Extract facial features from Google Vision API
    extract_features = (len(faces) > 1) or (len(faces) <= 0)
    landmarks = []
    angles = []
    bounding_box = []
    # Only extract features if one face detected or there is a face
    if extract_features == False:
        for face in faces:
            # Extract roll, tilt, pan angles and confidence
            print("Landmark Confidence: ", face.landmarking_confidence)
            
            # Extract bounding box
            for vertex in face.bounding_poly.vertices:
                bounding_box.append ([vertex.x, vertex.y])
            
            # Get the angles 
            angles.append(face.roll_angle)
            angles.append(face.tilt_angle)
            angles.append(face.pan_angle)
            
            # Extract the facial landmarks
            for landmark in face.landmarks:
                #print("Landmark type: ", landmark.type_)
                positions = [landmark.position.x, landmark.position.y, landmark.position.z]
                landmarks.append(positions)

            # Convert array to numpy array
            landmarks = np.array(landmarks)
            angles = np.array(angles)
            
            #print("Num Landmarks: ", len(landmarks))
            #print('face bounds: {}'.format(','.join(vertices)))
            # Print emotions
            #print ("Joy: ", face.joy_likelihood)
            #print ("Sad: ", face.sorrow_likelihood)
            #print ("Anger: ", face.anger_likelihood)
    else:
        print ("ERROR: More than one face or no faces detected for file: ")
        print ("Length: ", len(faces))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return landmarks, angles

# Calculate pairwise distances between landmarks
# Extract pairwise distances between landmarks
def calc_dist (landmarks, angles):
    pdist = sp.pdist(landmarks, metric='euclidean')

    # Find avg_landmark
    avg_landmark = np.average(landmarks, axis=0)

    # Extract distances between landmarks and average landmark point
    dist_to_avg = []
    for mark in landmarks:
        dist = sp.euclidean (mark, avg_landmark)
        dist_to_avg.append (dist)
    dist_to_avg = np.array(dist_to_avg)

    # Concatenate all the features (FINAL DIMS: (598,))
    facial_features = np.concatenate ((pdist, dist_to_avg, angles), axis=0)
    return facial_features
 
# Get a feature matrix for the example video
def get_feature_mat (video_name, start, end):
    mat = np.zeros(shape=(598,))
    if end == 0:
        stop = False
        i = 0
        while (not stop):
            path_to_file = video_name + "_" + str(i) + ".jpg"
            # break when cannot fine another frame
            if (not os.path.exists(path_to_file)):
                print("Could not find it: ", path_to_file)
                break
            print ("File ", i)
            file_name = os.path.abspath(path_to_file)
            faces, response = call_google_vision (file_name)
            landmarks, angles = extract_landmarks (faces, response)
            if (len(landmarks) == 0 or len(angles) == 0):
                return np.array([])
            facial_features = calc_dist (landmarks, angles)
            mat = np.vstack ((mat, facial_features))
            i += 1
    return mat


true_dir = "/home/conradli/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Clips/Truthful/"
lie_dir = "/home/conradli/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Clips/Deceptive/"
need_crop = [1, 2, 3, 6, 7, 8, 9, 11, 17, 21, 25, 26, 41, 42, 45, 48, 50, 51, 55, 56]
need_crop_lie = [5, 7, 10, 17, 20, 21, 22, 23, 24, 29, 31, 37, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
keep_lie = [32, 33, 34, 35, 36, 2, 3, 4, 8, 9, 14, 19, 11, 25, 28]
for i in range (1, 61):
    if i in keep_lie:
        number = "{0:0=3d}".format(i)
        mat = get_feature_mat (lie_dir + "trial_lie_" + number, 0, 0)
        if (mat.size == 0):
            continue
        mat = np.delete(mat, (0), axis=0)
        scaler = MinMaxScaler()
        mat  = scaler.fit_transform(mat)
        #print(mat)
        print(i, " ", mat.shape)
        np.save ("/home/conradli/HackTX20/training_data/lie/lie_" + number, mat)

#print("{0:0=3d}".format(3))

#mat = np.load ("/home/conradli/HackTX20/training_data/lie/lie_001.npy")
#print(mat.shape)
#scaler = MinMaxScaler()
#mat  = scaler.fit_transform(mat)
#print (mat)
#np.save ("/home/conradli/HackTX20/training_data/lie/lie_001.npy", mat)


# Main 
'''file_name = '/home/conradli/trial_lie_001_0.jpg'
faces, response = call_google_vision (file_name)
landmarks, angles = extract_landmarks (faces, response)
facial_features = calc_dist (landmarks, angles)
facial_features = np.append(facial_features, [1])
mat = np.zeros(shape=(599,))
mat = np.vstack ((mat, facial_features))
print(mat.shape)
mat = np.delete(mat, (0), axis=0)
print(mat.shape)
print(mat)
#print(facial_features)

if len (facial_features != 598):
    print ("ERROR: ", file_name)'''

#print(facial_features)
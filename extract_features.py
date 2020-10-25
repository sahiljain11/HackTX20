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
        print ("ERROR: More than one face or no faces detected for file: ", file_name)
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
                print(path_to_file)
                break
            file_name = os.path.abspath(path_to_file)
            faces, response = call_google_vision (file_name)
            landmarks, angles = extract_landmarks (faces, response)
            facial_features = calc_dist (landmarks, angles)
            mat = np.vstack ((mat, facial_features))
            i += 1
    return mat

'''
dir = "/home/conradli/HackTX20/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Clips/Truthful/"
mat = get_feature_mat (dir + "trial_truth_002", 0, 0, 0)
mat = np.delete(mat, (0), axis=0)
print(mat)
print(mat.shape)
np.save ("/home/conradli/HackTX20/truth_002",mat)
'''


#mat = np.load ("/home/conradli/HackTX20/training_data/truth/truth_002.npy")
#print(mat.shape)
#mat = np.delete(mat, mat.shape[1]-1, axis=1)
#print (mat)
#np.save ("/home/conradli/HackTX20/training_data/truth/truth_002.npy", mat)


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
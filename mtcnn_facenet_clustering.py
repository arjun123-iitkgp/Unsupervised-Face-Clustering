from keras_facenet import FaceNet
import glob
import dlib
import os
import cv2 
from sklearn.cluster import KMeans

embedder = FaceNet()

faces_folder_path = r'testfaces/'#containing image files to be processed

images = []
faces = 0 
delstore = []
facedet = []

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    detections = embedder.extract(img, threshold=0.95)
    images.append(img)
    print("num faces found : ",len(detections))
    faces = faces + len(detections)
    for i in range(len(detections)):
        delstore.append(detections[i]['embedding']) #generating 512d vectors for each detected face
        facedet.append((img,detections[i]['box']))  #storing the bounding box of each detected face


delstorevec = []
for i in delstore:
    delstorevec.append(dlib.vector(i))

#chinese whispher graph clustering
labels = dlib.chinese_whispers_clustering(delstorevec, 0.8)
#tune to achieve optimal number or clusters
num_classes = len(set(labels)) # Total number of clusters
print("Number of clusters: {}".format(num_classes))

#kmeans clustering with predefined number of clusters as per number of members in videos
kmeans = KMeans(n_clusters=2, random_state=0).fit(delstorevec)

#num_classes_chinese_whisper = len(set(labels))
labels_kmeans = kmeans.labels_

#labels_cw = labels
num_classes_k = len(set(labels_kmeans))

#creating output directories for clusters
output_folder = r'output_kmeans_cluster/'
for i in range(0, num_classes_k):
    output_folder_path = output_folder + '/output' + str(i)
    os.path.normpath(output_folder_path)
    os.makedirs(output_folder_path)

for i in range(len(labels_kmeans)):
    t = labels_kmeans[i] 
    output_folder_path = output_folder + '/output' + str(t) # Output folder for each cluster
    img = facedet[i][0]
    face = img[facedet[i][1][1]:(facedet[i][1][1]+facedet[i][1][3]),facedet[i][1][0]:(facedet[i][1][0]+facedet[i][1][2])]
    #please name facial boxes extracted accordingly
    file_path = os.path.join(output_folder_path,"face_"+str(t)+"_"+str(i))+".jpg"
    cv2.imwrite(file_path,face)
import os, time
import dlib
import numpy as np
from pyparsing import empty
from skimage import io
import cv2
from datetime import datetime
import multiprocessing
import time


# Globals
faces_folder_path = 'Images/'
dlib_frontal_face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
face_classifier_opencv = cv2.CascadeClassifier(os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml")


def timefn(fn, args):
    start = datetime.now()
    r = fn(*args)
    elapsed = datetime.now() - start
    print(fn.__name__ + " took ", elapsed)
    return r


def to_dlib_rect(w, h):
    return dlib.rectangle(left=0, top=0, right=w, bottom=h)

def to_rect(dr):
    #  (x, y, w, h)
    return dr.left(), dr.top(), dr.right()-dr.left(), dr.bottom()-dr.top()

def face_detector_opencv(image):
    # if image is not None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # else:
    #     print('empty frame')
    return face_classifier_opencv.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE)

def face_detector_dlib(image):
    bounds = dlib_frontal_face_detector(image, 0) 
    return list(map(lambda b: to_rect(b), bounds))


def get_face_encodings(face, bounds):

    start = datetime.now()
    faces_landmarks = [shape_predictor(face, face_bounds) for face_bounds in bounds]
    return [np.array(face_recognition_model.compute_face_descriptor(face, face_pose, 1)) for face_pose in faces_landmarks]


def get_face_matches(known_faces, face):
    return np.linalg.norm(known_faces - face, axis=1)


def find_match(known_faces, person_names, face):
    matches = get_face_matches(known_faces, face) # get a list of True/False
    min_index = matches.argmin()
    min_value = matches[min_index]
    if min_value < 0.58:
        return person_names[min_index]
    return 'Unknown'


def load_face_encodings(faces_folder_path):
    image_filenames = filter(lambda x: x.endswith('.jpeg'), os.listdir(faces_folder_path))
    image_filenames = sorted(image_filenames)
    person_names = [x[:-4] for x in image_filenames]

    full_paths_to_images = [faces_folder_path + x for x in image_filenames]
    known_faces = []


    for path_to_image in full_paths_to_images:
        face = io.imread(path_to_image)

        faces_bounds = dlib_frontal_face_detector(face, 0)

        if len(faces_bounds) != 1:
            print("Expected one and only one face per image: " + path_to_image + " - it has " + str(len(faces_bounds)))
            exit()

        face_bounds = faces_bounds[0]
        face_landmarks = shape_predictor(face, face_bounds)
        face_encoding = np.array(
            face_recognition_model.compute_face_descriptor(face, face_landmarks, 1)
        )


        known_faces.append(face_encoding)

    return known_faces, person_names

def find_webcam():
    for i in range(10):
        cap=cv2.VideoCapture(i)
        if  cap.isOpened():
            return i

def mark_attendance(name):
    with open ('attendancelist.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',') 
            namelist.append(entry[0])
        if name not in namelist and name!='Unknown':
            now=datetime.now()
            datestring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datestring} ')

def rescale_frame(frame, scale):    # works for image, video, live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def recognize_faces_in_video(known_faces, person_names,camId):
    cap = cv2.VideoCapture(camId)
    while True:
        ret, frame = cap.read()
        if not(ret):
            st = time.time()
            cap = cv2.VideoCapture(camId)
            print("time lost due to reinitialization : ",time.time()-st)
            continue
        face_rects = face_detector_opencv(frame)
        # print("Getting bounds took ", (datetime.now() - start))

        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]

            bounds = ([to_dlib_rect(w.item(), h.item())])  # int32 when opencv
            #bounds = ([to_dlib_rect(w, h)])
            face_encodings_in_image = get_face_encodings(face, bounds)
            if (face_encodings_in_image):
                match = find_match(known_faces, person_names, face_encodings_in_image[0])
                cv2.putText(frame, match, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                mark_attendance(match)
        
        frame = rescale_frame(frame, scale=.5)
        cv2.imshow("camera", frame)
        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# Process class
class Process(multiprocessing.Process):
    # global known_faces,person_names
    def __init__(self, camid,known_faces,person_names):
        super(Process, self).__init__()
        self.camid = camid
        self.known_faces=known_faces
        self.person_names=person_names
    def run(self):
        recognize_faces_in_video(self.known_faces, self.person_names,self.camid)




if __name__ == '__main__':
    cap1='rtsp:192.168.0.101'
    known_faces, person_names = load_face_encodings(faces_folder_path)
    p=Process(cap1,known_faces, person_names)
    # Create a new process and invoke the
    # Process.run() method
    p.start()
    cap2 = 0
    p=Process(cap2,known_faces, person_names)
    p.start()




#  import threading
# class camThread(threading.Thread):
#     def __init__(self, previewName, camID):
#         threading.Thread.__init__(self)
#         self.previewName = previewName
#         self.camID = camID
#     def run(self):
#         print ('Starting ' + self.previewName)
#         camPreview(self.previewName, self.camID)

# def camPreview(previewName, camID):
#     cv2.namedWindow(previewName)
#     cam = cv2.VideoCapture(camID)
#     if cam.isOpened():  # try to get the first frame
#         rval, frame = cam.read()
#     else:
#         rval = False

#     while rval:
#         cv2.imshow(previewName, frame)
#         rval, frame = cam.read()
#         key = cv2.waitKey(20)
#         if key == 27:  # exit on ESC
#             break
#     cv2.destroyWindow(previewName)

# Create two threads as follows
# thread1 = camThread('Camera 1', 'rtsp://admin:12345@192.168.0.118/H264?ch=1&subtype=0')
# thread2 = camThread('Camera 2', 'rtsp://admin:12345@192.168.0.117/H264?ch=1&subtype=0')
# thread1.start()
# thread2.start()

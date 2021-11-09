# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import datetime
import csv
import pandas as pd

totalcount=0
flag =0
conlist=[]
class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = r'C:\Users\Sresth\Desktop\opencv\faster_rcnn_inception_v2_coco_2018_01_28\frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture('ts69.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    
    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    current_day = str(datetime.datetime.now().day)
    current_month = str(datetime.datetime.now().month)
    current_year = str(datetime.datetime.now().year)
    current_time = str(datetime.datetime.now().hour)
    extension = ".avi"
    file_name = current_day+'_'+current_month+'_' + \
        current_year+'_'+current_time + extension
    # file_name='fot2.avi'
    result = cv2.VideoWriter(file_name, 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)

    while True:
        r, img = cap.read()
        #img = cv2.resize(img,(224,224),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

        # boxes, scores, classes, num = odapi.processFrame(img)
        try:
            boxes, scores, classes, num = odapi.processFrame(img)
        except:
            with open(r'C:\Users\Sresth\Desktop\opencv\count1.csv','a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(conlist)
            break
        count=0
        count1 =0
        flag=0
        # Visualization of the results of a detection.

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,255,0),2)
                cx=box[1] + (box[3]-box[1])/2
                cy=box[0] + (box[2]-box[0])/2
                centroid = (cx,cy)
                #print(cx)
                count +=1
                cv2.circle(img, (int(cx), int(cy)), 3, (0, 255,0),cv2.FILLED)
                if cy>150 and cy<152:
                    count1+=1
                    flag =1
                    totalcount+=1
                
        cv2.putText(img, 'Instantaneous Person Count'+str(count), (5, frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'Total Person Count'+str(totalcount), (5, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.putText(img, 'centroid'+str(centroid), (20, frame_height - 40),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.line(img, (0, 150),(800, 150), (60, 250, 255),2)
        conlist.append(count)
        print(totalcount)
        print(conlist)
        # define codec and create VideoWriter object
        # with open('/home/19110209/rcnn/count1.csv','a', encoding='UTF8') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(conlist)
        result.write(img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

video.release()
result.release()
    
# Closes all the frames
cv2.destroyAllWindows()
   
print("The video was successfully saved")
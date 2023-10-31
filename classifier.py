import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
# from maze_detector import clf
import sys

import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import csv
import sys
import time
import numpy as np
import cv2
from cv_bridge import CvBridge
import pickle

class MinimalVideoSubscriber(Node):
    
    # def knn(self):
        # self.dilatem = (2,2)
        # self.erodem = (3,3)
    #     imageDirectory = '/home/jayden/colcon_ws/src/team5_final/team5_final/2022Simgs/'
    #     # imageDirectory = '~/colcon_ws/src/team5_final/team5_final/2022Simgs/'
    #     imageDirectory2 = '/home/jayden/colcon_ws/src/team5_final/team5_final/2022Fimgs/'

    #     with open(imageDirectory + 'train.txt', 'r') as f:
    #         reader = csv.reader(f)
    #         lines = list(reader)

    #     # this line reads in all images listed in the file in GRAYSCALE, and resizes them to 33x25 pixels
    #     train = np.array([np.array(cv2.resize(cv2.imread(imageDirectory +lines[i][0]+".jpg"),(33,25))) for i in range(len(lines))])

    #     # print("shape", train.shape)
    #     n_samples, y_dim, x_dim, data_dim = train.shape
    #     satvals = np.zeros((n_samples, y_dim, x_dim))
    #     satvalsmasked = np.zeros((n_samples, y_dim, x_dim))

    #     #PUT IMAGE PROCESSING HERE
    #     #############################
    #     sats_threshold = 100
    #     for i in range(len(lines)):
    #         train[i] = cv2.cvtColor(train[i],cv2.COLOR_BGR2HSV)
    #         h, sats, v = cv2.split(train[i])
    #         temp = sats.copy()
    #         temp[np.where(temp>=sats_threshold)] = 255
    #         temp[np.where(temp<sats_threshold)] = 0
    #         # temp = sats[np.where(sats > 100)]
    #         satvalsmasked[i] = temp.astype(np.uint8)
    #         satvalsmasked[i] = cv2.dilate(satvalsmasked[i],self.dilatem)
    #         satvalsmasked[i] = cv2.erode(satvalsmasked[i],self.erodem)

    #     satvalsmasked = satvalsmasked.astype(np.uint8)
    #     contours = []
    #     for i in range(len(lines)):
    #         contour, h = cv2.findContours(satvalsmasked[i],cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    #         if len(contour) == 0:
    #             contours.append(None)
    #             continue

    #         largest_contour = max(contour, key=cv2.contourArea)
    #         contours.append(largest_contour)

    #     contours = np.array(contours)
    #     # print("CONTOURS", contours.shape)

    #     crop = np.zeros((n_samples, y_dim, x_dim))
    #     # print(satvals.shape)
    #     for i in range(len(lines)):
    #         # crop.append(0)
    #         if(contours[i] is None):
    #             continue
    #         x, y, w, h = cv2.boundingRect(contours[i])
    #         # print(satvals[i],satvals[i].shape, x,y,w,h)
    #         intermed = satvalsmasked[i,y:y+h,x:x+w]
    #         crop[i] = cv2.resize(intermed,(33,25))

    #     crop = crop.astype(np.uint8)
    #     #############################

    #     # here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
    #     # print("CROP", crop.shape)
    #     img_idx = 50
    #     cDraw = cv2.drawContours(train[img_idx], contours[img_idx],-1,(0,255,0),3)
    #     # cv2.imshow('c',cDraw)
    #     # cv2.imshow('s',satvalsmasked[img_idx])
    #     # cv2.imshow('scale',crop[img_idx])
    #     # cv2.waitKey()
    #     train_data = crop.flatten().reshape(len(lines), 33*25)
    #     # print(train.shape)
    #     # train_data = train.flatten().reshape(len(lines), 3* 27*22)
    #     train_data = train_data.astype(np.float32)
    #     # scaler = preprocessing.StandardScaler().fit(train_data)
    #     # train_data = scaler.transform(train_data)
    #     # read in training labels
    #     train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])


    #     with open(imageDirectory2 + 'train.txt', 'r') as f:
    #         reader = csv.reader(f)
    #         lines = list(reader)

    #     # this line reads in all images listed in the file in GRAYSCALE, and resizes them to 33x25 pixels
    #     train2 = np.array([np.array(cv2.resize(cv2.imread(imageDirectory2 +lines[i][0]+".png"),(33,25))) for i in range(len(lines))])
    #     n_samples, y_dim, x_dim, data_dim = train2.shape
    #     # print(train2.shape)


    #     #PUT IMAGE PROCESSING HERE
    #     #############################
    #     # train2 = train2[:,0:-3,3:-3]
    #     s = train2.shape
    #     # print(s)
    #     satvals = np.zeros((s[0],s[1],s[2]))
    #     satvalsmasked = np.zeros((s[0],s[1],s[2]))
    #     #PUT IMAGE PROCESSING HERE
    #     #############################
    #     for i in range(len(lines)):
            
    #         train2[i] = cv2.cvtColor(train2[i],cv2.COLOR_BGR2HSV)
    #         h,sats,v = cv2.split(train2[i])
    #         # cv2.imshow('sats',sats)
    #         # cv2.waitKey()
    #         temp = sats.copy()
    #         temp[np.where(temp>=80)] = 255
    #         temp[np.where(temp<80)] = 0
    #         # temp = sats[np.where(sats > 100)]
    #         satvalsmasked[i] = temp.astype(np.uint8)
    #         satvalsmasked[i] = cv2.dilate(satvalsmasked[i],self.dilatem)
    #         satvalsmasked[i] = cv2.erode(satvalsmasked[i],self.erodem)



    #     satvalsmasked = satvalsmasked.astype(np.uint8)
    #     contours = []
    #     for i in range(len(lines)):
    #         contour, h = cv2.findContours(satvalsmasked[i],cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    #         if len(contour) == 0:
    #             contours.append(None)
    #             continue

    #         largest_contour = max(contour, key=cv2.contourArea)
    #         contours.append(largest_contour)

    #     contours = np.array(contours)
    #     # print("CONTOURS", contours.shape)

    #     crop = np.zeros((n_samples, y_dim, x_dim))
    #     # print(satvals.shape)
    #     for i in range(len(lines)):
    #         # crop.append(0)
    #         if(contours[i] is None):
    #             continue
    #         x, y, w, h = cv2.boundingRect(contours[i])
    #         # print(satvals[i],satvals[i].shape, x,y,w,h)
    #         intermed = satvalsmasked[i,y:y+h,x:x+w]
    #         crop[i] = cv2.resize(intermed,(33,25))

    #     crop = crop.astype(np.uint8)

    #     #############################

    #     # print("TRAIN:", train_data.shape)

    #     # # here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
    #     train_data2 = crop.flatten().reshape(len(lines), 33*25)
    #     # train_data2 = train2.flatten().reshape(len(lines), 3* 27*22)
    #     train_data2 = train_data2.astype(np.float32)
    #     train_data = np.append(train_data,train_data2,axis=0)
    #     self.scaler = preprocessing.StandardScaler().fit(train_data)
    #     train_data = self.scaler.transform(train_data)
    #     # read in training labels
    #     train_labels2 = np.array([np.int32(lines[i][1]) for i in range(len(lines))])
    #     train_labels = np.append(train_labels,train_labels2,axis=0)

    #     # print(train_data.shape)
    #     # print(train_labels.shape)

    #     #lbfgs app works better with smaller dataasets
    #     # clf = MLPClassifier(hidden_layer_sizes=(33*25*3,825,825,75,6),solver='sgd',random_state=1,max_iter=2000, verbose=True, alpha=.0001, early_stopping=False, learning_rate='adaptive').fit(train_data,train_labels)
    #     # clf = MLPClassifier(hidden_layer_sizes=(33*25*3,825,25,6),solver='lbfgs',random_state=1,max_iter=2000, verbose=True, alpha=.1).fit(train_data,train_labels)

    #     clf = KNeighborsClassifier(2,n_jobs=-1)
    #     clf.fit(train_data,train_labels)
    #     print(clf.score(train_data,train_labels))
    #     with open(imageDirectory + 'test.txt', 'r') as f:
    #         reader = csv.reader(f)
    #         lines = list(reader)

    #     # this line reads in all images listed in the file in GRAYSCALE, and resizes them to 33x25 pixels
    #     test = np.array([np.array(cv2.resize(cv2.imread(imageDirectory +lines[i][0]+".jpg"),(33,25))) for i in range(len(lines))])
    #     n_samples, y_dim, x_dim, data_dim = test.shape
    #     # h, tests, v = cv2.split(test[i] for i in range(len(lines))) 
    #     # tests = np.array(tests)
    #     satvals = np.zeros((n_samples, y_dim, x_dim))
    #     satvalsmasked = np.zeros((n_samples, y_dim, x_dim))
    #     sats_threshold = 100
    #     for i in range(len(lines)):
    #         test[i] = cv2.cvtColor(test[i],cv2.COLOR_BGR2HSV)
    #         h, sats, v = cv2.split(test[i])
    #         temp = sats.copy()
    #         temp[np.where(temp>=sats_threshold)] = 255
    #         temp[np.where(temp<sats_threshold)] = 0
    #         # temp = sats[np.where(sats > 100)]
    #         satvalsmasked[i] = temp.astype(np.uint8)
    #         satvalsmasked[i] = cv2.dilate(satvalsmasked[i],self.dilatem)
    #         satvalsmasked[i] = cv2.erode(satvalsmasked[i],self.erodem)
    #     satvalsmasked = satvalsmasked.astype(np.uint8)
    #     contours = []
    #     for i in range(len(lines)):
    #         contour, h = cv2.findContours(satvalsmasked[i],cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    #         if len(contour) == 0:
    #             contours.append(None)
    #             continue

    #         largest_contour = max(contour, key=cv2.contourArea)
    #         contours.append(largest_contour)

    #     contours = np.array(contours)
    #     # print(contours.shape)

    #     crop = np.zeros((n_samples, y_dim, x_dim))
    #     # print(satvals.shape)
    #     for i in range(len(lines)):
    #         # crop.append(0)
    #         if(contours[i] is None):
    #             continue
    #         x, y, w, h = cv2.boundingRect(contours[i])
    #         # print(satvals[i],satvals[i].shape, x,y,w,h)
    #         intermed = satvalsmasked[i,y:y+h,x:x+w]
    #         crop[i] = cv2.resize(intermed,(33,25))

    #     crop = crop.astype(np.uint8)


    #     # #############################
    #     # crop = np.zeros((s[0],s[1],s[2]))

    #     # # print(satvals.shape)
    #     # for i in range(len(lines)):
    #     #     # crop.append(0)
    #     #     if(contour[i] == []):
    #     #         continue
    #     #     x,y,w,h = cv2.boundingRect(contour[i])
    #     #     # print(satvals[i],satvals[i].shape, x,y,w,h)
    #     #     intermed = satvals[i,y:y+h,x:x+w]
    #     #     crop[i] = cv2.resize(intermed,(33,25))

    #     # cv2.imshow('int',intermed[0])
    #     # cv2.imshow('img1',test[0])
    #     # cv2.imshow('img',crop[0])
    #     # cv2.waitKey(0)
    #     # print(sats.shape)

    #     # test = test[:,0:-3,3:-3,:]
    #     # here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
    #     test_data = crop.flatten().reshape(len(lines), 33*25)
    #     # test_data = test.flatten().reshape(len(lines), 3* 27*22)
    #     test_data = test_data.astype(np.float32)
    #     test_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])
    #     with open(imageDirectory2 + 'test.txt', 'r') as f:
    #         reader = csv.reader(f)
    #         lines = list(reader)

    #     # this line reads in all images listed in the file in GRAYSCALE, and resizes them to 33x25 pixels
    #     test2 = np.array([np.array(cv2.resize(cv2.imread(imageDirectory2 +lines[i][0]+".png"),(33,25))) for i in range(len(lines))])
    #     n_samples, y_dim, x_dim, data_dim = test2.shape
    #     # print('HERE',test2.shape)

    #     s = test2.shape
    #     satvals = np.zeros((s[0],s[1],s[2]))
    #     satvalsmasked = np.zeros((s[0],s[1],s[2]))
    #     #PUT IMAGE PROCESSING HERE
    #     #############################
    #     for i in range(len(lines)):
            
    #         test2[i] = cv2.cvtColor(test2[i],cv2.COLOR_BGR2HSV)
    #         h,sats,v = cv2.split(test2[i])
    #         # cv2.imshow('sats',sats)
    #         # cv2.waitKey()
    #         temp = sats.copy()
    #         temp[np.where(temp>=80)] = 255
    #         temp[np.where(temp<80)] = 0
    #         # temp = sats[np.where(sats > 100)]
    #         satvalsmasked[i] = temp.astype(np.uint8)
    #         satvalsmasked[i] = cv2.dilate(satvalsmasked[i],self.dilatem)
    #         satvalsmasked[i] = cv2.erode(satvalsmasked[i],self.erodem)

    #     satvalsmasked = satvalsmasked.astype(np.uint8)
    #     contours = []
    #     for i in range(len(lines)):
    #         contour, h = cv2.findContours(satvalsmasked[i],cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    #         if len(contour) == 0:
    #             contours.append(None)
    #             continue

    #         largest_contour = max(contour, key=cv2.contourArea)
    #         contours.append(largest_contour)

    #     contours = np.array(contours)
    #     # print(contours.shape)

    #     crop = np.zeros((n_samples, y_dim, x_dim))
    #     # print(satvals.shape)
    #     for i in range(len(lines)):
    #         # crop.append(0)
    #         if(contours[i] is None):
    #             continue
    #         x, y, w, h = cv2.boundingRect(contours[i])
    #         # print(satvals[i],satvals[i].shape, x,y,w,h)
    #         intermed = satvalsmasked[i,y:y+h,x:x+w]
    #         crop[i] = cv2.resize(intermed,(33,25))

    #     crop = crop.astype(np.uint8)

    #     # # test2 = test2[:,0:-3,3:-3,:]
    #     # print(test2.shape)
    #     # # here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
    #     test_data2 = crop.flatten().reshape(len(lines),33*25)
    #     # a = test2.shape
    #     # # test_data2 = test2.flatten().reshape(len(lines), 3* 27*22)
    #     test_data2 = test_data2.astype(np.float32)
    #     test_data = np.append(test_data,test_data2,axis=0)
    #     # scaler = preprocessing.StandardScaler().fit(test_data)
    #     test_data = self.scaler.transform(test_data)
    #     # print(test_data.shape)
    #     # read in testing labels
    #     test_labels2 = np.array([np.int32(lines[i][1]) for i in range(len(lines))])
    #     test_labels = np.append(test_labels,test_labels2,axis=0)
    #     print(clf.score(train_data,train_labels))
    #     print('Training Data', clf.predict(train_data))
    #     print(train_labels)

    #     print(clf.score(test_data,test_labels))
    #     print('Testing Data',clf.predict(test_data))
    #     print(test_labels)
    #     print('CLF HERE',clf)
    #     pickle.dump(clf,open('model.sav','wb'))
    #     return clf


    def __init__(self):		
        # Creates the node.
        super().__init__('minimal_video_subscriber')

        # Set Parameters
        self.declare_parameter('show_image_bool', False)
        self.declare_parameter('window_name', "Raw Image")

        #Determine Window Showing Based on Input
        self._display_image = bool(self.get_parameter('show_image_bool').value)

        # Declare some variables
        self._titleOriginal = self.get_parameter('window_name').value # Image Window Title	
        if(self._display_image):
        # Set Up Image Viewing
            cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE ) # Viewing Window
            cv2.moveWindow(self._titleOriginal, 50, 50) # Viewing Window Original Location
    
        #Declare that the minimal_video_subscriber node is subcribing to the /camera/image/compressed topic.
        self._video_subscriber = self.create_subscription(
                CompressedImage,
                '/camera/image/compressed',
                self._image_callback,
                1)
        self._video_subscriber # Prevents unused variable warning.

        self.clf = pickle.load(open('model.sav','rb'))
        self.scaler = pickle.load(open('/home/jayden/colcon_ws/src/team5_final/team5_final/scaler.sav','rb'))


    def classify(self, img):
        ##CHECK PUBLISHER FOR CLASS HERE
        clf = self.clf
        scaler = self.scaler
        # clf = pickle.load(open('model.sav','rb'))
        # print(np.shape(self.img))
        # img = cv2.resize(img,(33,25,3))
        # print(np.shape(img))
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,sats,v = cv2.split(hsv)
        sats = cv2.resize(sats,(33,25))
        # cv2.imshow('sats',sats)
        # cv2.waitKey()
        temp = sats.copy()
        temp[np.where(temp>=80)] = 255
        temp[np.where(temp<80)] = 0
        # temp = sats[np.where(sats > 100)]
        satvalsmasked = temp.astype(np.uint8)



        satvalsmasked = satvalsmasked.astype(np.uint8)
        satvalsmasked = cv2.dilate(satvalsmasked,(2,2))
        satvalsmasked = cv2.erode(satvalsmasked,(3,3))
        contours = []
        # for i in range(len(lines)):
        contour, h = cv2.findContours(satvalsmasked,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        if len(contour) == 0:
            contours.append(None)


        else:
            largest_contour = max(contour, key=cv2.contourArea)
            contours.append(largest_contour)

        contours = np.array(contours)
        # print("CONTOURS", contours.shape)

        crop = np.zeros(( 33, 25))
        # print(satvals.shape)
        # for i in range(len(lines)):
            # crop.append(0)
        if(contours[0] is None):
            #PUBLISH -1
            #
            return 0

        x, y, w, h = cv2.boundingRect(contours[0])
        # print(satvals[i],satvals[i].shape, x,y,w,h)
        intermed = satvalsmasked[y:y+h,x:x+w]
        crop = cv2.resize(intermed,(33,25))

        crop = crop.astype(np.uint8)
        # self.show_image(crop)
        self.processedImage = crop.flatten().reshape(1, 33*25)
        # train_data2 = train2.flatten().reshape(len(lines), 3* 27*22)
        self.processedImage = self.processedImage.astype(np.float32)
        # self.scaler.fit(self.processedImage)
        self.processedImage = scaler.transform(self.processedImage)
        #PUBLISH

        return clf.predict(self.processedImage)

        #Print and stuff
        # print(circles)
        # return circles
        

    def _image_callback(self, CompressedImage):	
        # The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"
        self.img = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
        print(np.shape(self.img))
        imgtwo = self.img
        # print(self.classify(imgtwo))
        # self.img = CvBridge
        if(self._display_image):
            # Display the image in a window
            self.show_image(self.img)	

    def get_image(self):
        return self._displayFrame



    def show_image(self, img):
        # if (self._circles is not None):
        #     self._circles = np.uint16(np.around(self._circles))
        #     for i in self._circles[-1,:]:
        #         cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)

        # cv2.imshow('frame',frame)
        cv2.imshow(self._titleOriginal, img)
        # Cause a slight delay so image is displayed
        self._user_input=cv2.waitKey(10) #Use OpenCV keystroke grabber for delay.

    def get_user_input(self):
        return self._user_input

class FlagSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.flag = False
        self.subscription = self.create_subscription(
            Float32,
            '/sign_req',
            self.listener_callback,
            1
        )
        self.subscription

    
    def listener_callback(self, msg):
        self.flag = msg.data == 1
        # print(self.flag)       # do something
    
    def getFlag(self):
        return self.flag

def main():
    settings = None

    rclpy.init()
    imageSubscriber = MinimalVideoSubscriber()
    flagSubscriber = FlagSubscriber() #TODO
    node = rclpy.create_node('classifier')
    classPublisher = node.create_publisher(Float32,'/sign',5)
    # null_msg = Float32MultiArray()
    # null_msg = [float(-1), float(-1)]
    # print("subscribe ready")
    # rclpy.spin(position_subscriber)
    # r_theta = Float32MultiArray()

    # print("subscriber made")
    # qos = QoSProfile(depth=10)
    # pub = node.create_publisher(Float32MultiArray, '/r_theta', 5)
    print('Classifier Ready')
    try:
        # print('try')
        while rclpy.ok():
            rclpy.spin_once(flagSubscriber) # Trigger callback processing.
            # print('not hung')
            # print(flagSubscriber.getFlag())
            if flagSubscriber.getFlag():
                print("request received")
                rclpy.spin_once(imageSubscriber) # Trigger callback processing.
                print('img') 
                pub = Float32()
                pub.data = float(imageSubscriber.classify(imageSubscriber.img))
                classPublisher.publish(pub)
                flagSubscriber.flag = False
            # print("center: ", angle_subscriber.get_center())
            # print("ranges: ", laser_subscriber.get_ranges())
            
    
    except Exception as e:
        print(e)

    finally:
        print('hi')


if __name__ == '__main__':
    main()

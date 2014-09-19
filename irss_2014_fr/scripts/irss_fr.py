#!/usr/bin/env python
import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import os

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback)
    self.pathpt1 = '/home/skel/FR_IRSS/Real_Scenario/image_'
    self.pathpt2 = '.jpg'
    self.sequence = 1
    
  def callback(self,data):
    try:
        check = rospy.get_param("/check_for_client")
    except:
        return
    if (check == 0):
        return
    if (self.sequence > 20):
        os.system("rm -r /home/skel/FR_IRSS/Persons")
        #os.system("python /home/skel/FR_IRSS/saveFacesintoFolders.py")
        #print 'EKEI PERA'
        #os.system("python /home/skel/FR_IRSS/winnerFace.py ")
        os.system("python /home/skel/FR_IRSS/script.py")
        os.system("gwenview /home/skel/FR_IRSS/winnerFace.jpg")
        time.sleep(10)
        rospy.signal_shutdown("Done")
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
      print e
    
    path = self.pathpt1 + str(self.sequence) + self.pathpt2
    print path
    time.sleep(0.5)
    try :
        cv2.imwrite(path, cv_image);
        self.sequence += 1
        print "image saved at " + path
    except:
        print"image not saved"
        #self.sequence -= 1
    
    
    #(rows,cols,channels) = cv_image.shape
    #if cols > 60 and rows > 60 :
    #  cv2.circle(cv_image, (50,50), 10, 255)

    #cv2.imshow("Image window", cv_image)
    #cv2.waitKey(3)

    #try:
    #  self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    #except CvBridgeError, e:
    #  print e

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

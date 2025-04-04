#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

def publish_image():
    rospy.init_node('image_publisher', anonymous=True)
    
    image_pub = rospy.Publisher('/camera/image', Image, queue_size=10)
    rate = rospy.Rate(10)  # Adjust the publish rate if needed
    
    # Initialize CvBridge
    bridge = CvBridge()

    # Load the PNG image using OpenCV
    image_path = '/home/tarun/Downloads/op.png'  # Replace with your image path
    cv_image = cv2.imread(image_path)

    if cv_image is None:
        rospy.logerr("Failed to load image. Check the file path.")
        return

    while not rospy.is_shutdown():
        # Create the header with the frame ID "camera_link"
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_link"

        # Convert OpenCV image to ROS Image message
        image_message = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        image_message.header = header

        # Publish the image
        image_pub.publish(image_message)

        rospy.loginfo("Published image to /camera/image topic")
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_image()
    except rospy.ROSInterruptException:
        pass

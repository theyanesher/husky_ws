#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from cv_bridge import CvBridge

# Lists to store x and y coordinates
x_coords = []
y_coords = []

# Initialize the CvBridge for image conversion
bridge = CvBridge()

# Create a ROS publisher for the image
image_pub = None

# Callback function for the topic subscriber
def path_callback(data):
    global x_coords, y_coords, image_pub

    # Clear previous data
    x_coords = []
    y_coords = []

    # Extract position information from the path
    for pose in data.poses:
        x = pose.pose.position.x
        y = pose.pose.position.y
        x_coords.append(x)
        y_coords.append(y)

    # Plot the path
    plt.clf()  # Clear the current figure
    plt.plot(x_coords, y_coords, marker='o', linestyle='-')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Global Plan Path')
    plt.grid(True)

    # Convert the Matplotlib plot to an OpenCV image
    plt.draw()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(canvas.get_width_height()[::-1] + (3,))

    # Convert the OpenCV image to a ROS Image message
    ros_image = bridge.cv2_to_imgmsg(image, "rgb8")

    # Publish the image to the ROS topic
    image_pub.publish(ros_image)

# Main function to initialize the ROS node and subscribe to the topic
def listener():
    global image_pub

    # Initialize the ROS node
    rospy.init_node('path_listener', anonymous=True)
    
    # Subscribe to the global plan topic
    rospy.Subscriber("/locomotor/global_plan", Path, path_callback)

    # Create the image publisher
    image_pub = rospy.Publisher("/global_plan_image", Image, queue_size=10)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import tf
import numpy as np

def publish_paths():
    rospy.init_node('path_publisher', anonymous=True)

    # Create a list of publishers for markers (one per path)
    marker_publishers = []
    num_paths = 25  # Adjust based on your number of paths
    for i in range(num_paths):
        topic_name = f'/diffusion_path_marker_{i+1}'
        marker_pub = rospy.Publisher(topic_name, Marker, queue_size=10)
        marker_publishers.append(marker_pub)

    # Load all x, y sets from the .npy file
    all_xy_values = np.array([0.7, -0.1]) + np.load('/home/tarun/husky_ws/src/husky_pure_pursuit/src/tejas_sample_trajs.npy') / np.array([100, 100])  # Replace with the actual path to your .npy file

    # Fixed orientation for all poses
    quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)  # roll, pitch, yaw

    # Pre-defined list of colors (RGBA) for the paths
    colors = [
        (1.0, 0.0, 0.0, 1.0),  # Red
        (0.0, 1.0, 0.0, 1.0),  # Green
        (0.0, 0.0, 1.0, 1.0),  # Blue
        (1.0, 1.0, 0.0, 1.0),  # Yellow
        (0.0, 1.0, 1.0, 1.0),  # Cyan
        (1.0, 0.0, 1.0, 1.0),  # Magenta
        (0.5, 0.5, 0.5, 1.0)   # Gray
    ]

    while not rospy.is_shutdown():
        for set_idx, xy_values in enumerate(all_xy_values):
            marker = Marker()
            marker.header.frame_id = "camera_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = f"path_{set_idx + 1}"
            marker.id = set_idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.02  # Line width (adjust as needed)

            # Set a unique color for each path, cycling through predefined colors
            r, g, b, a = colors[set_idx % len(colors)]
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = a

            # Add points to the marker, rotating by 90 degrees
            for (x, y) in xy_values:
                # Rotate the point (x, y) by 90 degrees
                x_rotated = -y
                y_rotated = x

                point = PoseStamped()
                point.pose.position.x = x_rotated
                point.pose.position.y = y_rotated
                point.pose.position.z = -0.3  # Set z position

                marker.points.append(point.pose.position)

            # Publish the marker for the current path
            marker_publishers[set_idx].publish(marker)
            rospy.loginfo(f"Published marker for path {set_idx + 1}")

        rospy.sleep(1)  # Adjust publish rate as needed

if __name__ == '__main__':
    try:
        publish_paths()
    except rospy.ROSInterruptException:
        pass

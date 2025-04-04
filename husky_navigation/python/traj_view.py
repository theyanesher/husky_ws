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
    all_xy_values =  np.load('/home/tarun/husky_ws/src/husky_pure_pursuit/src/tejas_sample_trajs.npy')  # Replace with the actual path to your .npy file
    temp = all_xy_values[:,:,0].copy()
    all_xy_values[:,:,0] = all_xy_values[:,:,1]
    all_xy_values[:,:,1] = -temp
    all_xy_values = np.array([.04, 0]) + all_xy_values / np.array([25, 30])

    # Fixed orientation for all poses
    quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)  # roll, pitch, yaw

    # Pre-defined list of light colors (RGBA) for the paths except the third one
    light_colors = [
        (0.8, 0.5, 0.5, 1.0),  # Light red
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

            # Set different thickness for diffusion_path_marker_3 (set_idx == 2)
            if set_idx == 2:
                marker.scale.x = 0.02  # Thicker line for /diffusion_path_marker_3
                # Set color to green for path 3
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0  # Full opacity
            else:
                marker.scale.x = 0.004  # Thinner line for all other paths
                # Use light colors for other paths
                r, g, b, a = light_colors[set_idx % len(light_colors)]
                marker.color.r = r
                marker.color.g = g
                marker.color.b = b
                marker.color.a = a

            # Add points to the marker
            for (x, y) in xy_values:
                point = PoseStamped()
                point.pose.position.x = x
                point.pose.position.y = y
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

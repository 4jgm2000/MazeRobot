import os
import select
import sys
import rclpy
from rclpy.node import Node
from numpy import isnan
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
import time

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 1.75 #2.84

COLLISION_DISTANCE = 0.45 # closest distance for an obstacle

# TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

class LaserSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self._ranges = None
        self._angle_min = None
        self._angle_max = None
        self._ancle_inc = None
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            qos_profile
        )
        self.subscription

    
    def listener_callback(self, msg):
        self._ranges = list(msg.ranges)
        self._angle_min = float(msg.angle_min)
        self._angle_max = float(msg.angle_max)
        self._angle_inc = float(msg.angle_increment)
            
    def get_ranges(self):
        return self._ranges

    def get_values(self):
        return (self._ranges, self._angle_min, self._angle_max, self._angle_inc)


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)
    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def local_min(objects):
    return [obj for i, obj in enumerate(objects)
            if ((i == 0) or (objects[i - 1][2] >= obj[2]))
            and ((i == len(objects) - 1) or (obj[2] < objects[i+1][2]))]


def main():
    settings = None
    rclpy.init()

    laser_subscriber = LaserSubscriber()
    null_msg = Float32MultiArray()
    null_msg.data = [float(-1)]
    
    obstacles = Float32MultiArray()

   
    node = rclpy.create_node('get_object_range')
    pub = node.create_publisher(Float32MultiArray, '/obstacles', 5)
    try:
        while rclpy.ok():
            rclpy.spin_once(laser_subscriber) # Trigger callback processing

            ranges, angle_min, angle_max, angle_inc = laser_subscriber.get_values()
            angle_min *= 180.0 / 3.14159
            angle_max *= 180.0 / 3.14159
            angle_inc *= 180.0 / 3.14159

            angle_step = 2
            angles = range(-110, 110, angle_step)

            objects_in_range = []
            # print('hhahahaha')
            for angle in angles:
                index = round(angle * len(ranges)/360.0)
                rangeVal = round(ranges[index],7)
                if not isnan(rangeVal) and rangeVal < COLLISION_DISTANCE:
                    objects_in_range.append([angle, index, rangeVal])
            # print('oir: ',objects_in_range)
            if len(objects_in_range) == 0:
                pub.publish(null_msg)
                continue
            elif len(objects_in_range) == 1:
                closest_angles = [float(int(objects_in_range[0][0]) + objects_in_range[0][2])]
            else:
                x = local_min(objects_in_range)
                # print("x: ", x)
                closest_angles = [float(int(i[0])+round(i[2],4)) for i in x] # angle.rangeVal

            print('CA: ',closest_angles)
            if len(closest_angles) == 0:
                pub.publish(null_msg)
                continue

            obstacles.data = [float(a) for a in closest_angles]
            pub.publish(obstacles)
            # time.sleep(0.2)
    
    except Exception as e:
        print(e)

    finally:
        pub.publish(null_msg)


if __name__ == '__main__':
    main()
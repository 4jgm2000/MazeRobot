from logging import critical
from math import pi, modf
import os
import select
import sys

from enum import Enum

import numpy as np
# from Rotation_Script import update_Odometry
import rclpy
from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import time
from dataclasses import dataclass

BURGER_MAX_LIN_VEL = 0.18
BURGER_MAX_ANG_VEL = 1.75 #2.84

stop_robot_msg = Twist()
stop_robot_msg.linear.x = 0.0
stop_robot_msg.linear.y = 0.0
stop_robot_msg.linear.z = 0.0
stop_robot_msg.angular.x = 0.0
stop_robot_msg.angular.y = 0.0
stop_robot_msg.angular.z = 0.0

reverse_robot_msg = Twist()
reverse_robot_msg.linear.x = -0.05
reverse_robot_msg.linear.y = 0.0
reverse_robot_msg.linear.z = 0.0
reverse_robot_msg.angular.x = 0.0
reverse_robot_msg.angular.y = 0.0
reverse_robot_msg.angular.z = 0.0

turn_robot_msg = Twist()
turn_robot_msg.linear.x = 0.0
turn_robot_msg.linear.y = 0.0
turn_robot_msg.linear.z = 0.0
turn_robot_msg.angular.x = 0.0
turn_robot_msg.angular.y = 0.0
turn_robot_msg.angular.z = -0.5

drive_robot_msg = Twist()
drive_robot_msg.linear.x = 1.0
drive_robot_msg.linear.y = 0.0
drive_robot_msg.linear.z = 0.0
drive_robot_msg.angular.x = 0.0
drive_robot_msg.angular.y = 0.0
drive_robot_msg.angular.z = 0.0

# TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

@dataclass
class Point:
    x: float
    y: float
    z: float

class State(Enum):
    SIGN_CLASSIFY = 1
    ROTATION = 2
    LINEAR_MOTION = 3
    GOAL_STATE = 4

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class ObjectsSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self._obstacles = []
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/obstacles',
            self.listener_callback,
            1
        )
        self.subscription

    def listener_callback(self, msg):
        self._obstacles = msg.data
        # do something
    
    def get_obstacles(self):

        if self._obstacles == []:
            return self._obstacles
        angles = []
        ranges = []
        for i in self._obstacles:
            ran,ang = modf(i)
            angles.append(ang)
            ranges.append(abs(ran))
        return ranges,angles

class SignClassifierSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self._sign = -1.0
        self.subscription = self.create_subscription(
            Float32,
            '/sign',
            self.listener_callback,
            1
        )
        self.subscription

    def listener_callback(self, msg):
        self._sign = msg.data
        # do something
    
    def get_sign(self):
        return self._sign

class OdometrySubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self._odometry = []
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.listener_callback,
            1
        )
        self.subscription
        
        self.Init = True
        self.Init_pos = Point(0,0,0)
        self.globalPos = Point(0,0,0)

    def listener_callback(self, msg):
        self._odometry = msg
        # print(self._odometry.pose.pose.position)
        self.update_Odometry()
        # do something
    
    def get_odometry(self):
        return self._odometry
    
    def get_global_pose(self):
        return (self.globalPos, self.globalAng)
    
    def update_Odometry(self):
        Odom = self._odometry

        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            # print('please be once')
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            # self.Init_pos = 
            # self.Init_pos
            # self.globalPos = position
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
            return

        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        #We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang
        # print('glo: ',self.globalPos)
        # print('pos: ',position)
        # print('init', self.Init_pos)

def angle_adjust(angle):
    while angle > pi:
        angle -= 2*pi
    while angle < -1*pi:
        angle += 2*pi
    
    return angle


def main():
    settings = None

    rclpy.init()
    objects_subscriber = ObjectsSubscriber()
    odometry_subscriber = OdometrySubscriber()
    sign_subscriber = SignClassifierSubscriber()
    
    curr_state = State.LINEAR_MOTION
    
    node = rclpy.create_node('go_to_goal')
    vel_pub = node.create_publisher(Twist, '/cmd_vel', 5)

    node2 = rclpy.create_node('request_sign')
    req_pub = node2.create_publisher(Float32, '/sign_req', 5)

    target_heading = 0
    target_heading_reached = True

    speed_const = 0.75
    omega_const = 1.0
    obs_o_const = 1.25

    reinit_odom = False

    prev_state = State.GOAL_STATE
    print("Ready")
    try:
        while rclpy.ok():
            vel = 0
            omega = 0
            # print('huh')
            rclpy.spin_once(objects_subscriber) # Trigger callback processing.
            rclpy.spin_once(odometry_subscriber) # Trigger callback processing.
            # print('not odom')
            # req_msg = Float32()
            # req_msg.data = float(0)
            # req_pub.publish(req_msg)

            ranges, obstacles = objects_subscriber.get_obstacles()
            global_pos, global_angle = odometry_subscriber.get_global_pose()

            global_angle = angle_adjust(global_angle)

            # if curr_state != prev_state:
            #     print("state:", curr_state)
            #     prev_state = curr_state

            if curr_state == State.LINEAR_MOTION:
                # print("State: linear")
                print("ranges, obstacles", ranges, obstacles)

                vel = BURGER_MAX_LIN_VEL * speed_const
                if obstacles[0] == -1:
                    # No obstacles detected
                    # Drive straight
                    vel = BURGER_MAX_LIN_VEL * speed_const
                    omega = 0
                else:
                    critical_angle = min(obstacles, key=abs) # min magnitude of the obstacles
                    critical_angle_range = ranges[obstacles.index(critical_angle)]

                    print('crit angle: ', critical_angle, critical_angle_range)

                    if abs(critical_angle) < 15 and target_heading_reached:
                        if critical_angle_range < 0.35:
                            print("Backing up")
                            vel_pub.publish(reverse_robot_msg)
                            continue
                        elif abs(critical_angle) > 2:
                            print("facing wall")
                            target_heading_reached = False
                            target_heading = -1*pi/180.0*critical_angle
                            curr_state = State.ROTATION
                            continue
                        else:
                            curr_state = State.SIGN_CLASSIFY
                            continue
                    # else:
                    #     # Wall Follow
                    #     print("Wall follow")
                    #     error = (abs(critical_angle) - 90)
                    #     omega = error * BURGER_MAX_ANG_VEL / 90.0
                    #     if critical_angle < 0:
                    #         omega *= -1

                drive_robot_msg.linear.x = float(vel)
                drive_robot_msg.angular.z = float(omega)
                vel_pub.publish(drive_robot_msg)   
                    
            elif curr_state == State.SIGN_CLASSIFY:
                # print("State: classify")
                # Obstacles detected
                # print('OBSTACLE')
                vel_pub.publish(stop_robot_msg)

                # critical_angle = min(obstacles, key=abs) # min magnitude of the obstacles
                # critical_angle_range = ranges[obstacles.index(critical_angle)]
                # # print('crit angle: ', critical_angle)
                # critical_angle_rad = (critical_angle * pi / 180.0)
                sign = None
                print("Sign in Front")
                req_msg = Float32()
                req_msg.data = float(1)
                req_pub.publish(req_msg)
                rclpy.spin_once(sign_subscriber)
                sign = sign_subscriber.get_sign()
                
                target_heading_reached = False
                reinit_odom = True

                if sign is None:
                    print("NO SIGN SCANNED")
                elif sign == 0:
                    print("sign: empty wall")
                    target_heading = pi
                elif sign == 1:
                    print("sign: left turn")
                    target_heading = pi/2
                elif sign == 2:
                    print("sign: right turn")
                    target_heading = -pi/2
                elif sign == 3:
                    print("sign: do not enter")
                    target_heading = pi
                elif sign == 4:
                    print("sign: stop")
                    target_heading = pi
                elif sign == 5:
                    print("sign: goal")
                    curr_state = State.GOAL_STATE
                
                curr_state = State.ROTATION
            
            elif curr_state == State.ROTATION:
                print("State: rotation")
                target_heading = angle_adjust(target_heading)
                
                theta_error = target_heading - global_angle
                theta_error = angle_adjust(theta_error)

                # print('theta_e: ', theta_error, theta_error*180.0/pi)
                # theta_sign = -1 if theta_error < 0 else 1

                target_heading_reached = abs(theta_error) < (2.0 * pi/180.0)
                if target_heading_reached and reinit_odom:
                    odometry_subscriber.Init = True
                    reinit_odom = False
                    curr_state = State.LINEAR_MOTION
                    continue

                omega = theta_error * BURGER_MAX_ANG_VEL * omega_const
                omega = np.clip(omega, -1 * BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
                # print('omega,' , omega)
                # omega = 0 if abs(omega) < (0.075 * BURGER_MAX_ANG_VEL) else omega

                turn_robot_msg.angular.z = float(omega)
                vel_pub.publish(turn_robot_msg)

            elif curr_state == State.GOAL_STATE:
                print("GOAL REACHED! :)")
                vel_pub.publish(stop_robot_msg)
                break
            
            time.sleep(0.2)

    except Exception as e:
        print(e)

    finally:
        vel_pub.publish(stop_robot_msg)


if __name__ == '__main__':
    main()
    
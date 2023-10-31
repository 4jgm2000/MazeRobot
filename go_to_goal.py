
from logging import critical
from math import pi
import os
import select
import sys

import numpy as np
# from Rotation_Script import update_Odometry
import rclpy
from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import time
from dataclasses import dataclass

BURGER_MAX_LIN_VEL = 0.18
BURGER_MAX_ANG_VEL = 1.75 #2.84

# TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

@dataclass
class Point:
    x: float
    y: float
    z: float

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
        return self._obstacles

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
    

def main():
    settings = None

    rclpy.init()
    objects_subscriber = ObjectsSubscriber()
    odometry_subscriber = OdometrySubscriber()
    
    # print("subscribe ready")
    # rclpy.spin(position_subscriber)
    
    # print("subscriber made")
    # qos = QoSProfile(depth=10)
    node = rclpy.create_node('go_to_goal')
    pub = node.create_publisher(Twist, '/cmd_vel', 5)

    waypoints = [Point(1.5, 0, 0), Point(1.5, 1.4, 0), Point(0, 1.4, 0)]
    waypoint_idx = 0

    stop_robot_msg = Twist()
    stop_robot_msg.linear.x = 0.0
    stop_robot_msg.linear.y = 0.0
    stop_robot_msg.linear.z = 0.0

    stop_robot_msg.angular.x = 0.0
    stop_robot_msg.angular.y = 0.0
    stop_robot_msg.angular.z = 0.0

    turn_right_msg = Twist()
    turn_right_msg.linear.x = 0.0
    turn_right_msg.linear.y = 0.0
    turn_right_msg.linear.z = 0.0

    turn_right_msg.angular.x = 0.0
    turn_right_msg.angular.y = 0.0
    turn_right_msg.angular.z = BURGER_MAX_ANG_VEL*0.5

    speed_const = 3
    omega_const = 2
    obs_o_const = 1.25

    try:
        while rclpy.ok():

            if waypoint_idx == len(waypoints):
                # All goals reached
                break

            rclpy.spin_once(objects_subscriber) # Trigger callback processing.
            rclpy.spin_once(odometry_subscriber) # Trigger callback processing.

            obstacles = objects_subscriber.get_obstacles()
            global_pos, global_angle = odometry_subscriber.get_global_pose()

            while global_angle > pi:
                global_angle -= 2*pi
            while global_angle < -1*pi:
                global_angle += 2*pi

            # print('global: ',global_pos)

            curr_target = waypoints[waypoint_idx]

            vel = None
            omega = None
            # print(obstacles[0])
            if obstacles[0] == -1:
                # No obstacles detected
                # print('no ob')
                # Compute Vectors
                v0 = np.array([curr_target.x, curr_target.y]) - np.array([global_pos.x, global_pos.y])
                # v1 = np.array([np.cos(global_angle), np.sin(global_angle)])

                # Calculate Linear Velocity
                # print(v0)y
                # if waypoint_idx % 2 == 0:
                #     dist_sign = -1 if v0[0] < 0  else 1
                # else:
                #     dist_sign = -1 if v1[1] < 0 else 1
               
                dist_error = np.linalg.norm(v0)
                print("dist_e: ", dist_error)
                if dist_error < 0.05:
                    # Goal Found
                    pub.publish(stop_robot_msg)
                    # time.sleep(1)
                    waypoint_idx += 1
                    print(waypoints[waypoint_idx])

                    continue

                # print('error: ', error)
                vel = dist_error * BURGER_MAX_LIN_VEL * speed_const
                vel = np.clip(vel, -1*BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
                vel = vel if abs(vel) > 0.15 * BURGER_MAX_LIN_VEL else 0

                #NEW
                theta_v0 = np.arctan2(v0[1],v0[0])
                
                print('theta v0: ', theta_v0*180.0/pi, ' || theta rob: ', global_angle*180.0/pi)
                theta_error = theta_v0 - global_angle
                while theta_error > pi:
                    theta_error -= 2*pi
                while theta_error < -1*pi:
                    theta_error += 2*pi
                
                print('theta_e', theta_error*180.0/pi)
                # theta_sign = -1 if theta_error < 0 else 1

                omega = theta_error * BURGER_MAX_ANG_VEL * omega_const
                omega = np.clip(omega, -1 * BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
                # print('omega,' , omega)
                omega = 0 if abs(omega) < (0.1 * BURGER_MAX_ANG_VEL) else omega
                # omega = omega * theta_sign
                # print('omega,', omega)
                # if omega != 0:
                #     vel = 0
                vel = 0 if abs(omega) > 0.15 * BURGER_MAX_ANG_VEL else vel
            else:
                # Obstacles detected
                print('OBSTACLE')
                critical_angle = min(obstacles, key=abs) # min magnitude of the obstacles
                print('crit angle: ', critical_angle)
                critical_angle = (critical_angle * pi / 180.0)
                vel = abs(critical_angle)* 2 / pi * BURGER_MAX_LIN_VEL 
                
                theta_error = (pi/2 - critical_angle) if critical_angle > 0 else (-pi/2 - critical_angle)
                print('theta_error', theta_error)
                # theta_error = 0 if abs(theta_error) < .1 else theta_error``
                omega =  -1 * (theta_error) * BURGER_MAX_ANG_VEL * obs_o_const
                omega = np.clip(omega, -1 * BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL) 
                omega = 0 if abs(omega) < (0.15 * BURGER_MAX_ANG_VEL) else omega
                vel = 0 if abs(omega) > 0.15 * BURGER_MAX_ANG_VEL else vel



            twist = Twist()

            
            twist.linear.x = float(vel) # 
            twist.linear.y = 0.0
            twist.linear.z = 0.0

            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = float(omega)

            print('vel: ', vel, 'omega: ', omega)
            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        pub.publish(stop_robot_msg)


if __name__ == '__main__':
    main()
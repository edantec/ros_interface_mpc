#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from ros_interface_mpc.msg import InputMessage
from rclpy.qos import QoSProfile

from inputs import get_gamepad
import numpy as np

class InputPublisher(Node):

    def __init__(self):
        super().__init__('input_publisher')
        qos_profile = QoSProfile(depth=10)
        
        self.publisher = self.create_publisher(InputMessage, 'input', qos_profile)
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.vlin = [0.0, 0.0]
        self.walk = False

    def timer_callback(self):
        events = get_gamepad()
        
        for event in events:
            if (event.code == "ABS_Y"):
                self.vlin[0] = float((128 - event.state) * 0.01)
            if (event.code == "ABS_X"):
                self.vlin[1] = float((128 - event.state) * 0.01)
            if (event.code == "BTN_EAST" and event.state == 1):
                self.walk = not(self.walk)
        
        #self.get_logger().info(str(self.vlin))
        #self.get_logger().info(str(self.walk))
        msg = InputMessage()
        msg.linear_vel = self.vlin
        msg.yaw_vel = float(0)
        msg.event = self.walk
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    input_publisher = InputPublisher()

    rclpy.spin(input_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    input_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import numpy as np

import rclpy
from rclpy.node import Node

# from crazyflie_interfaces.msg import LogDataGeneric
# from geometry_msgs.msg import PoseStamped

from sensor_msgs.msg import Joy

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

class VisualizationNode(Node):

    def __init__(self):
        super().__init__('vis')

        self.cf = "cf18"

        self.setParamsService = self.create_client(SetParameters, "/crazyflie_server/set_parameters")

        # self.publisher = self.create_publisher(
                # PoseStamped, "{}/frontnet_targetpos_typed".format(self.cf), 10)

        self.subscription1 = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10)


    def joy_callback(self, msg: Joy):
        # the expected configuration is
        # vars: ["frontnet.targetx", "frontnet.targety", "frontnet.targetz", "frontnet.targetyaw"]

        # self.get_logger().info('I heard: "%s"' % msg)
        if msg.buttons[2] == 1:   # blue button
            # self.get_logger().info("Blue button")
            param_name = "cf18.params.frontnet.start"
            value = 1
            param_type = ParameterType.PARAMETER_INTEGER
            param_value = ParameterValue(type=param_type, integer_value=int(value))
            req = SetParameters.Request()
            req.parameters = [Parameter(name=param_name, value=param_value)]
            self.setParamsService.call_async(req)

        if msg.buttons[6] == 1:   # back (land) button
            param_name = "cf18.params.frontnet.start"
            value = 0
            param_type = ParameterType.PARAMETER_INTEGER
            param_value = ParameterValue(type=param_type, integer_value=int(value))
            req = SetParameters.Request()
            req.parameters = [Parameter(name=param_name, value=param_value)]
            self.setParamsService.call_async(req)


def main(args=None):
    rclpy.init(args=args)

    node = VisualizationNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from ghost_manager_interfaces.srv import SetParam

class GaitParameterSetterNode(Node):
    def __init__(self):
        super().__init__('gait_parameter_setter_node')

        # Create client for set_param service
        self.param_client = self.create_client(SetParam, '/set_param')
        while not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/set_param service not available, waiting again...')
        self.param_req = SetParam.Request()

        # Subscribe to gait parameters topic
        self.parameter_subscriber = self.create_subscription(
            Float32MultiArray,
            '/gait_parameters',
            self.parameter_callback,
            10
        )

        # Parameter names corresponding to the indices in the received data
        self.param_names = ['PCW_RETRACT', 'Y_HSPLAY']

    def parameter_callback(self, msg):
        parameters = msg.data

        if len(parameters) != len(self.param_names):
            self.get_logger().error(f"Received parameter list of incorrect length: {len(parameters)}")
            return

        # Set the parameters using the set_param service
        for name, value in zip(self.param_names, parameters):
            self.set_param(name, [value])

        self.get_logger().info(f"Set parameters: {dict(zip(self.param_names, parameters))}")

    def set_param(self, param_name, value, planner=False):
        self.param_req.param.name = param_name
        self.param_req.param.val = value
        self.param_req.param.planner = planner
        future = self.param_client.call_async(self.param_req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is not None:
            self.get_logger().info('%s' % response.result_str)
        else:
            self.get_logger().error('Failed to receive response from set_param service')

def main(args=None):
    rclpy.init(args=args)
    gait_parameter_setter_node = GaitParameterSetterNode()
    rclpy.spin(gait_parameter_setter_node)
    gait_parameter_setter_node.destroy_node()
    rclpy.shutdown()

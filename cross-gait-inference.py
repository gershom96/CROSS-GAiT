import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import numpy as np
from collections import deque

# Import your model class
from networks.fused_model_ import FusionModelWithRegression

class CrossGaitInferenceNode(Node):
    def __init__(self):
        super().__init__('gait_parameter_inference_node')

        self.bridge = CvBridge()
        self.latest_image = None

        self.time_series_buffer = deque(maxlen=100)

        self.image_subscriber = self.create_subscription(
            Image,
            '/argus/ar0234_front_left/image_raw',
            self.image_callback,
            10)

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/mcu/state/imu',
            self.imu_callback,
            100)

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/mcu/state/jointURDF',
            self.joint_state_callback,
            100)

        self.latest_imu_msg = None
        self.latest_joint_state_msg = None

        # Publisher
        self.parameter_publisher = self.create_publisher(
            Float32MultiArray,
            '/gait_parameters',
            10
        )

        regressor_path = './checkpoints/regressor_checkpoint_epoch_2_step_final.pth'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fusion_model_with_regression = FusionModelWithRegression().to(self.device)
        checkpoint = torch.load(regressor_path, map_location=self.device)
        self.fusion_model_with_regression.load_state_dict(checkpoint['model_state_dict'])
        self.fusion_model_with_regression.eval()

        self.param_names = ['PCW_RETRACT', 'Y_HSPLAY']

        self.time_series_mean = np.array([
            -9.7528e-03,  7.1942e-02, -9.8050e+00, -1.3312e-03, -4.1957e-04, -2.5726e-02,
            -1.9685e+00,  1.6953e+01, -4.1611e+00,  1.8878e+01, -1.0623e+00,  1.6011e+01,
            -3.7153e+00,  1.9238e+01, -1.3624e+01, -1.3926e+01,  1.3605e+01,  1.2424e+01
        ])
        self.time_series_std = np.array([
            1.5537,  0.9568,  3.3148,  0.1399,  0.1181,  0.3017,
            13.1651, 17.3949, 13.9258, 19.9675, 13.2737, 17.1195,
            13.4313, 19.9113, 20.9347, 20.9335, 20.7401, 19.3905
        ])
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])

        self.timer = self.create_timer(0.5, self.inference_callback)

        self.step_height = 0.14
        self.hip_splay = 0.05

        self.max_d_step_height = 0.01
        self.max_d_hip_splay = 0.01

        self.max_step_height = 0.3
        self.min_step_height = 0.03

        self.max_hip_splay = 0.15
        self.min_hip_splay = 0.05

    def image_callback(self, msg):
        self.latest_image = msg

    def imu_callback(self, msg):
        self.latest_imu_msg = msg
        self.update_time_series_buffer()

    def joint_state_callback(self, msg):
        self.latest_joint_state_msg = msg
        self.update_time_series_buffer()

    def update_time_series_buffer(self):
        if self.latest_imu_msg is None or self.latest_joint_state_msg is None:
            return

        angular_velocity = self.latest_imu_msg.angular_velocity
        linear_acceleration = self.latest_imu_msg.linear_acceleration

        angular_velocity = np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z])
        linear_acceleration = np.array([linear_acceleration.x, linear_acceleration.y, linear_acceleration.z])

        joint_efforts = np.array(self.latest_joint_state_msg.effort)[:12]

        time_series_sample = np.concatenate([linear_acceleration, angular_velocity, joint_efforts])
        self.time_series_buffer.append(time_series_sample)

        self.latest_imu_msg = None
        self.latest_joint_state_msg = None

    def process_image(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return None

        height, width, _ = cv_image.shape
        mid = width // 2
        col_start = mid - 112
        col_end = mid + 112
        row_start = height - 224
        row_end = height + 1

        cv_image = cv_image[row_start:row_end, col_start:col_end, :]
        cv_image = cv_image.astype(np.float32) / 255.0

        cv_image = (cv_image - self.image_mean) / self.image_std

        image_tensor = torch.from_numpy(cv_image).permute(2, 0, 1)

        return image_tensor

    def inference_callback(self):
        if self.latest_image is None or len(self.time_series_buffer) < 100:
            self.get_logger().info('Waiting for image and sufficient time series data...')
            return

        image = self.process_image(self.latest_image)
        if image is None:
            return

        time_series_array = np.array(self.time_series_buffer)
        time_series_array = time_series_array.T

        time_series_mean = self.time_series_mean.reshape(-1, 1)
        time_series_std = self.time_series_std.reshape(-1, 1)
        time_series_array = (time_series_array - time_series_mean) / time_series_std

        time_series_tensor = torch.from_numpy(time_series_array).float()

        images = torch.unsqueeze(image, 0).to(self.device)
        time_series_tensor = torch.unsqueeze(time_series_tensor, 0).to(self.device)

        with torch.no_grad():
            output = self.fusion_model_with_regression(images, time_series_tensor)

        output = output.cpu().numpy().tolist()[0]
        print(output)
        step_height, hip_splay = self.dynamic_window_gait_set(output)

        # Prepare the message
        params_msg = Float32MultiArray()
        params_msg.data = [step_height, hip_splay]

        # Publish the parameters
        self.parameter_publisher.publish(params_msg)

        self.get_logger().info(f"Published parameters: Step Height: {step_height}, Hip Splay: {hip_splay}")

    def dynamic_window_gait_set(self, params):

        predicted_step_height = params[0]
        predicted_hip_splay = params[1]

        # Calculate differences
        d_step_height = predicted_step_height - self.step_height
        d_hip_splay = predicted_hip_splay - self.hip_splay

        # Limit the changes
        if abs(d_step_height) > self.max_d_step_height:
            d_step_height = np.sign(d_step_height) * self.max_d_step_height

        if abs(d_hip_splay) > self.max_d_hip_splay:
            d_hip_splay = np.sign(d_hip_splay) * self.max_d_hip_splay

        # Update parameters
        self.step_height += d_step_height
        self.hip_splay += d_hip_splay

        # Enforce bounds
        self.step_height = np.clip(self.step_height, self.min_step_height, self.max_step_height)
        self.hip_splay = np.clip(self.hip_splay, self.min_hip_splay, self.max_hip_splay)

        # Round off to two decimal places
        self.step_height = round(self.step_height, 2)
        self.hip_splay = round(self.hip_splay, 2)

        return self.step_height, self.hip_splay

def main(args=None):
    rclpy.init(args=args)
    cross_gait_inference_node = CrossGaitInferenceNode()
    rclpy.spin(cross_gait_inference_node)
    cross_gait_inference_node.destroy_node()
    rclpy.shutdown()

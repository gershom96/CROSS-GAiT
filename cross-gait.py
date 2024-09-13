import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import numpy as np
from collections import deque
from ghost_manager_interfaces.srv import EnsureMode, SetParam
from std_msgs.msg import UInt32

# Import your model class
from networks.fused_model_ import FusionModelWithRegression

class CrossGait(Node):
    def __init__(self):
        super().__init__('param_bag_recorder')

        self.cli = self.create_client(SetParam, '/set_param')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/set_param service not available, waiting again...')
        self.req = SetParam.Request()

        self.mode_client = self.create_client(EnsureMode, 'ensure_mode')
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('EnsureMode service not available, waiting again...')
        self.mode_req = EnsureMode.Request()

        self.param_client = self.create_client(SetParam, 'set_param')
        while not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('SetParam service not available, waiting again...')
        self.param_req = SetParam.Request()

        self.ensure_mode("control_mode", 180)
        self.ensure_mode("action", 2)  

        # self.hill_pub = self.create_publisher(UInt32, '/command/setHill', 10)
        # self.set_Mode = UInt32()
        # self.set_Mode.data = 1
        # self.unset_Mode = UInt32()
        # self.unset_Mode.data = 0
        # self.hill_pub.publish(self.set_Mode)

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
        # for name, value in zip(self.param_names, output):
        #     self.set_param(name, [value]) 

        
        self.get_logger().info(f"Set parameters: {dict(zip(self.param_names, output))}")

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

    def ensure_mode(self, field_name, valdes):
        self.mode_req.field = field_name
        self.mode_req.valdes = valdes
        future = self.mode_client.call_async(self.mode_req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is not None:
            self.get_logger().info('%s' % response.result_str)
        else:
            self.get_logger().error('Failed to receive response from ensure_mode service')

    def dynamic_window_gait_set(self, params):
        
        d_step_height = abs(params[0]-self.step_height)
        d_hip_splay = abs(params[1]-self.hip_splay)
        
        step_height = self.step_height
        hip_splay = self.hip_splay

        limit_step = False
        limit_hip = False

        if( d_step_height > self.max_d_step_height ):
            step_height = step_height+self.max_d_step_height
            limit_step = True

        if( d_hip_splay > self.max_d_hip_splay ):
            hip_splay = hip_splay+self.max_d_hip_splay
            limit_hip = True
        
        if( step_height > self.max_step_height ):
            step_height = self.max_step_height
            limit_step = True        
        elif( step_height < self.min_step_height ):
            step_height = self.min_step_height
            limit_step = True
        if ( hip_splay > self.max_hip_splay ):
            hip_splay = self.max_hip_splay
            limit_hip = True
        elif ( hip_splay < self.min_hip_splay ):
            hip_splay = self.max_hip_splay
            limit_hip = True

        if(not limit_hip and not limit_step):
            step_height = round(params[0],2)
            hip_splay = round(params[1],2)
        
        self.step_height = step_height
        self.hip_splay = hip_splay

def main(args=None):
    rclpy.init(args=args)
    cross_gait = CrossGait()
    rclpy.spin(cross_gait)
    cross_gait.destroy_node()
    rclpy.shutdown()

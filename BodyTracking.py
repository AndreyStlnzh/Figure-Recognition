import time
import math
import json
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np

from keypoints import *


class LogWriter:
    detection_result_size = 33

    def __init__(self, streaming=0):
        self.log = {'3D': [], '2D': []}
        self.streaming = streaming

    def add_record(self, detection_result, detection_result_2d, time: str, frame_num):
        # if not detection_result:
        #     return
        landmarks_list = [
            {
                'x': str(landmark[0]),
                'y': str(landmark[1]),
                'z': str(landmark[2]),
            }
            for landmark in detection_result
        ]

        landmarks_list_2d = [
            {
                'x': str(landmark[0]),
                'y': str(landmark[1]),
            }
            for landmark in detection_result_2d
        ]
        self.log['3D'].append({'landmarks': landmarks_list, 'time': time, 'frame_num': frame_num})
        self.log['2D'].append({'landmarks': landmarks_list_2d, 'time': time, 'frame_num': frame_num})


    def save_log(self, name):
        with open(name, 'w') as file:
            json.dump(self.log, file)


class BodyTracking:
    def __init__(self, video_path=None, real_time=False):
        self.video_path = video_path
        self.logger = LogWriter()
        self.real_time = real_time
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.body_param = sl.BodyTrackingParameters()
        self.positional_tracking_parameters = sl.PositionalTrackingParameters()
        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.viewer = gl.GLViewer()
        self.bodies = sl.Bodies()
        self.image = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()
        self.display_resolution = None
        self.image_scale = None

        self.f_x, self.f_y = 0, 0
        self.c_x, self.c_y = 0, 0
        self.cv_width = 1000
        self.cv_height = 500

        self.previous_spine_keypoint = None
        self.change_previous = True

        self.rect = None

        self.pause = False

        self.configure()

    def configure(self):
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        self.init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
        # init_params.coordinate_units = sl.UNIT.CENTIMETER
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        # init_params.camera_fps = 60

         # If applicable, use the SVO given as parameter
        # Otherwise use ZED live stream
        if self.video_path:
            filepath = self.video_path
            print("Using SVO file: {0}".format(filepath))
            self.init_params.svo_real_time_mode = self.real_time # False
            self.init_params.set_from_svo_file(filepath)



        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters
        # Focal length of the left eye in pixels
        self.f_x, self.f_y = calibration_params.left_cam.fx, calibration_params.left_cam.fy
        self.c_x, self.c_y = calibration_params.left_cam.cx, calibration_params.left_cam.cy

        #  If the camera is static, uncomment the following line to have better performances
        self.positional_tracking_parameters.set_as_static = True
        self.zed.enable_positional_tracking(self.positional_tracking_parameters)

        # Enable tracking to detect objects across time and space
        self.body_param.enable_tracking = True                # Track people across images flow
        # Optimize the person joints position, requires more computations
        self.body_param.enable_body_fitting = True            # Smooth skeleton move
        self.body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        self.body_param.body_format = sl.BODY_FORMAT.BODY_38  # Choose the BODY_FORMAT you wish to use
        # self.body_param.enable_segmentation = False

        # Enable Object Detection module
        self.zed.enable_body_tracking(self.body_param)

        self.body_runtime_param.detection_confidence_threshold = 40
        # self.body_runtime_param.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
        # self.runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD


        # Get ZED camera information
        camera_info = self.zed.get_camera_information()
        calibration_params = camera_info.camera_configuration.calibration_parameters
        # Focal length of the left eye in pixels
        self.f_x, self.f_y = calibration_params.left_cam.fx, calibration_params.left_cam.fy
        self.c_x, self.c_y = calibration_params.left_cam.cx, calibration_params.left_cam.cy

        # 2D viewer utilities
        self.display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, self.cv_width), \
                                           min(camera_info.camera_configuration.resolution.height, self.cv_height))
        
        
        self.image_scale = [self.display_resolution.width / camera_info.camera_configuration.resolution.width
                    , self.display_resolution.height / camera_info.camera_configuration.resolution.height]
        
        self.viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, self.body_param.enable_tracking, self.body_param.body_format)


    def calculate_angle_2d(self, three_point_xy):
        a = np.array(three_point_xy[0]) # First
        b = np.array(three_point_xy[1]) # Mid
        c = np.array(three_point_xy[2]) # End
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
            
        # if angle > 180.0:
        #     angle = 360 - angle

        return angle 


    def calculate_angle_3d(self, three_point_xy):
        a = np.array(three_point_xy[0]) # First
        b = np.array(three_point_xy[1]) # Mid
        c = np.array(three_point_xy[2]) # End
        
        vector_1 = a - b
        vector_2 = c - b
        radians = np.arccos((vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1] + vector_1[2] * vector_2[2]) \
                            / ((math.sqrt(vector_1[0] ** 2 + vector_1[1] ** 2 + vector_1[2] ** 2)) \
                            * (math.sqrt(vector_2[0] ** 2 + vector_2[1] ** 2 + vector_2[2] ** 2))))
        
        angle = np.abs(radians * 180.0 / np.pi)
        
        return angle


    def euclidian_distance(self, first_p, second_p):
        return math.sqrt((second_p[0] - first_p[0]) ** 2 + (second_p[1] - first_p[1]) ** 2 + (second_p[2] - first_p[2]) ** 2)


    def draw_trace(self, image, points, width_koef, height_koef):
        # angle_right = self.calculate_angle_3d([keypoint[right_marks[0]], keypoint[right_marks[1]], keypoint[right_marks[2]]])
        # angle_left = self.calculate_angle_3d([keypoint[left_marks[0]], keypoint[left_marks[1]], keypoint[left_marks[2]]])

        for i in range(len(points) - 1):
            if points[i] is not None and points[i+1] is not None:
                image = cv2.line(
                    image, 
                    [int(points[i][0] * width_koef), int(points[i][1] * height_koef)],
                    [int(points[i+1][0] * width_koef), int(points[i+1][1] * height_koef)],
                    (0, 255, 0), 
                    3,
                )

    def run(self, logging=False):
        frame_rate = 30
        frameRateMs = frame_rate / 1000
        timestamp = 0

        prev_points = [None] * 60
        point = RIGHT_WRIST

        while self.viewer.is_available():

            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                frame_num = self.zed.get_svo_position()
                
                # Retrieve left image
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution)
                # Retrieve bodies
                self.zed.retrieve_bodies(self.bodies, self.body_runtime_param)

                body_array = self.bodies.body_list
                if body_array:
                    child_body = body_array[0]
                            
                    keypoint_2d = child_body.keypoint_2d
                    keypoint = child_body.keypoint

                    cv2.putText(self.image.get_data(), str(frame_num), \
                        [20, 20], 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                    bb = child_body.bounding_box_2d
                    # cv2.rectangle(self.image.get_data(), (int(bb[0][0]*self.image_scale[0]), int(bb[0][1]*self.image_scale[1])), \
                    #             (int(bb[2][0]*self.image_scale[0]), int(bb[2][1]*self.image_scale[1])), (102,255,102), 6)
                
                    timestamp = frame_num / frameRateMs

                    self.logger.add_record(keypoint, keypoint_2d, round(timestamp, 3), frame_num)

                    
                    prev_points.pop(0)
                    if keypoint_2d[point][0] >= 0:
                        prev_points.append(keypoint_2d[point])
                    else:
                        prev_points.append(None)


                self.draw_trace(self.image.get_data(), prev_points, self.image_scale[0], self.image_scale[1])
                if frame_num > 50:
                    print(prev_points[-1])
                    pass


                self.viewer.update_view(self.image, self.bodies) 
                image_left_ocv = self.image.get_data()
                cv_viewer.render_2D(image_left_ocv, self.image_scale, self.bodies.body_list, self.body_param.enable_tracking, self.body_param.body_format)
                cv2.imshow("ZED | 2D View", image_left_ocv)
                cv2.waitKey(10)
                
        self.viewer.exit()
        self.image.free(sl.MEM.CPU)
        self.zed.close()

        if logging:
            self.logger.save_log(f'sample_{str(time.time())}.json')


if __name__ == "__main__":
    # body_tracking = BodyTracking("16.02.2024\HD720_SN22035595_14-01-17.svo", real_time=True)
    body_tracking = BodyTracking("16.02.2024\VGA_SN22035595_13-58-54.svo", real_time=True)
    # body_tracking = BodyTracking("16.02.2024\VGA_SN22035595_13-59-32.svo", real_time=True)
    body_tracking.run(logging=False)
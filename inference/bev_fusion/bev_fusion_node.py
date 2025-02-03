#!/usr/bin/env python3
from collections import deque
from functools import lru_cache, partial

import rclpy
from rclpy import qos
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import quaternion_from_euler, quaternion_multiply, quaternion_matrix, rotation_matrix
from tf2_msgs.msg import TFMessage
from std_srvs.srv import Empty
from sensor_msgs.msg import PointCloud2, CameraInfo, CompressedImage, Image
from message_filters import TimeSynchronizer, Subscriber
from bev_fusion import libpybev
import numpy as np
import cv2
import ros2_numpy
import tensor

from autoware_perception_msgs.msg import DetectedObjects, DetectedObject, ObjectClassification, DetectedObjectKinematics, Shape

NUSCENES_CLASSES = (
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone")


AUTOWARE_CLASSES = {"UNKNOWN": 0,
                    "CAR": 1,
                    "TRUCK": 2,
                    "BUS": 3,
                    "TRAILER": 4,
                    "MOTORCYCLE": 5,
                    "BICYCLE": 6,
                    "PEDESTRIAN": 7,}

CAMERAS = ["camera_front_wide",
           "camera_front_right",
           "camera_front_left",
           "camera_back",
           "camera_back_left",
           "camera_back_right",

           ]

class BEVFusion(Node):
    def __init__(self) -> None:
        super().__init__('BEVFusion')
        self.image_time = None
        self.camera2lidar = None
        self.threshold = self.declare_parameter('threshold', 0.1).value
        self.world_frame = self.declare_parameter('world_frame', 'map').value
        image_topic = self.declare_parameter('image_topic', 'image/compressed').value
        self.pkg_path = self.declare_parameter('pkg_path', '.').value
        self.core = None
        self.nuscenes_to_autoware = [0] * len(NUSCENES_CLASSES)
        autoware_classes = {v: k for k, v in AUTOWARE_CLASSES.items()}
        
        classes_mapping_info = "Nuscenes \t->\t Autoware\n"
        for i, c in enumerate(NUSCENES_CLASSES):
            self.nuscenes_to_autoware[i] = AUTOWARE_CLASSES.get(c.upper(), 0)
            classes_mapping_info += f"{i}.{c} \t->\t {self.nuscenes_to_autoware[i]}.{autoware_classes[self.nuscenes_to_autoware[i]]}\n"
        self.get_logger().info(classes_mapping_info)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.objects_pub = self.create_publisher(DetectedObjects, 'objects', 10)

        self.points_buffer = deque(maxlen=5)
        self.points_sub = self.create_subscription(PointCloud2, '/lidars/points_fused', self.points_callback, qos.qos_profile_sensor_data)

        camera_info_subs = [Subscriber(self, CameraInfo, f'/{camera}/camera_info', qos_profile=rclpy.qos.qos_profile_sensor_data)
                            for camera in CAMERAS]
        self.camera_info_sub = TimeSynchronizer(camera_info_subs, queue_size=len(CAMERAS)*2)
        self.camera_info_sub.registerCallback(self.camera_info_callback)
        self.camera_info_subs = camera_info_subs
        
        image_msg_type = CompressedImage if 'compressed' in image_topic else Image
        image_subs = [Subscriber(self, image_msg_type, f'/{camera}/{image_topic}', qos_profile=rclpy.qos.qos_profile_sensor_data)
                      for camera in CAMERAS]
        self.image_sub = TimeSynchronizer(image_subs, queue_size=len(CAMERAS)*2)
        self.image_sub.registerCallback(self.image_callback)

        self.saving_data = False
        self.save_data_srv = self.create_service(Empty, '~/save_data', self.save_data_callback)

        self.get_logger().info(self.get_name() + " is ready")

    def save_data_callback(self, request, response):
        self.saving_data = True
        return response

    def init_core(self, image_width=1600, image_height=900):
        self.images = np.zeros((1, 6, image_height, image_width, 3), dtype=np.uint8)

        model = self.declare_parameter('model', 'resnet50').value  # resnet50/resnet50int8/swint
        precision = self.declare_parameter('precision', 'fp16').value  # fp16/int8

        core = libpybev.load_bevfusion(
                f"{self.pkg_path}/model/{model}/build/camera.backbone.plan",
                f"{self.pkg_path}/model/{model}/build/camera.vtransform.plan",
                f"{self.pkg_path}/model/{model}/lidar.backbone.xyz.onnx",
                f"{self.pkg_path}/model/{model}/build/fuser.plan",
                f"{self.pkg_path}/model/{model}/build/head.bbox.plan",
                precision,
                image_width, image_height)
        
        if core is None:
            self.get_logger().error("Failed to create core")

        img_aug_matrix = np.array([[   0.48,    0.,      0.,    -32.  ],
                                   [   0.,      0.48,   0.,   -176.  ],
                                   [   0.,      0.,      1.,      0.  ],
                                   [   0.,      0.,      0.,      1.  ]], dtype=np.float32)
        self.img_aug_matrix = np.stack([img_aug_matrix] * 6)[None]
        camera_intrinsics = np.array([[image_width, 0, image_width, 0],
                                      [0, image_height, image_height, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], dtype=np.float32)
        self.camera_intrinsics = np.stack([camera_intrinsics] * 6)[None]

        core.set_capacity_points(100000 * self.points_buffer.maxlen)
        self.get_logger().info("Core initialized")
        self.core = core

    @lru_cache
    def get_static_tf(self, parent_frame, child_frame):
        return self.get_tf(parent_frame, child_frame, rclpy.time.Time())

    def get_tf(self, parent_frame, child_frame, timestamp, timeout=rclpy.time.Duration()):
        tf_msg = self.tf_buffer.lookup_transform(parent_frame, child_frame, timestamp, timeout)
        m = quaternion_matrix([tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w])
        m[:3, 3] = [tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z]
        return m

    def points_callback(self, points_msg):
        if self.core is None:
            return

        points = ros2_numpy.numpify(points_msg)
        points = points.view((points.dtype[0], len(points.dtype.names))).astype(np.float32)  # x, y, z, intensity
        points = np.hstack((points, np.zeros((len(points), 1), dtype=points.dtype)))  # add time column
        now = rclpy.time.Time.from_msg(points_msg.header.stamp)
        try:
            world2lidar_tf = self.get_tf(self.world_frame, points_msg.header.frame_id, now,
                                         rclpy.time.Duration(seconds=0.1))
        except TransformException as ex:
            self.get_logger().warn(f"{ex}")
            return
        points_in_world = points[:, :3] @ world2lidar_tf[:3, :3].T + world2lidar_tf[:3, 3]
        points_in_world = np.hstack((points_in_world, points[:, 3:]))
        self.points_buffer.append((now, points_in_world))

        if self.image_time is None:
            return
        
        try:
            world2lidar_tf_at_camera_time = self.get_tf(self.world_frame, points_msg.header.frame_id, self.image_time,
                                            rclpy.time.Duration(seconds=0.1))
        except TransformException as ex:
            self.get_logger().warn(f"{ex}")
            return

        for t, points in self.points_buffer:
            dt =  now - t
            points[:, -1] = dt.nanoseconds / 1e9

        densified_points_in_world = np.concatenate([d[1] for d in self.points_buffer], axis=0)
        lidar2world_tf = np.linalg.inv(world2lidar_tf)
        densified_points = densified_points_in_world[:, :3] @ lidar2world_tf[:3, :3].T + lidar2world_tf[:3, 3]
        densified_points = np.hstack((densified_points, densified_points_in_world[:, 3:])).astype(np.float16)
        # print('points:', densified_points.shape, densified_points.dtype)

        if self.camera2lidar is None:
            self.camera2lidar = np.stack([np.eye(4, dtype=np.float32)] * 6)[None]
            for i, camera in enumerate(CAMERAS):
                self.camera2lidar[0, i] = self.get_static_tf(points_msg.header.frame_id, camera)  # should called lidar2camera
            self.lidar2image = self.camera_intrinsics @ np.linalg.inv(self.camera2lidar)
        
        camera_motion = lidar2world_tf @ world2lidar_tf_at_camera_time
        lidar2image = self.lidar2image @ np.linalg.inv(camera_motion.astype(np.float32))
        self.core.update(self.camera2lidar, self.camera_intrinsics, lidar2image, self.img_aug_matrix)

        if self.saving_data:
            self.save_data(lidar2image, densified_points)
            self.saving_data = False

        boxes = self.core.forward(self.images, densified_points)
        self.publish_boxes(boxes.astype(float), points_msg.header.stamp, points_msg.header.frame_id)


    def camera_info_callback(self, *camera_info_msgs):
        if self.core is None:
            # assume all cameras have the same resulution
            for camera_info_msg in camera_info_msgs:
                if camera_info_msg.width != camera_info_msgs[0].width or camera_info_msg.height != camera_info_msgs[0].height:
                    self.get_logger().error(f"Cameras have different resolutions between {camera_info_msg.header.frame_id} and {camera_info_msgs[0].header.frame_id}")
                    return

            self.init_core(camera_info_msgs[0].width, camera_info_msgs[0].height)
            for i, camera_info_msg in enumerate(camera_info_msgs):
                self.camera_intrinsics[0, i, :3, :3] = np.array(camera_info_msg.k).reshape((3, 3))

            # destroy camera_info_subs
            for sub in self.camera_info_subs:
                self.destroy_subscription(sub.sub)
            del self.camera_info_subs
            del self.camera_info_sub


    def image_callback(self, *image_msgs):
        if self.core is None:
            return

        self.image_time = rclpy.time.Time.from_msg(image_msgs[0].header.stamp)
        for i, image_msg in enumerate(image_msgs):
            if isinstance(image_msg, CompressedImage):
                np_arr = np.frombuffer(image_msg.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                image = image[..., ::-1] # BGR to RGB
            elif isinstance(image_msg, Image):
                image = ros2_numpy.numpify(image_msg)
            else:
                raise ValueError(f"Unknown image type {type(image_msg)}")
            self.images[0, i] = image

    def publish_boxes(self, boxes, stamp, frame_id):
        # np.set_printoptions(3, suppress=True, linewidth=300)
        # print(boxes[:10])

        objects = DetectedObjects()
        objects.header.stamp = stamp
        objects.header.frame_id = frame_id

        for box in boxes:  # [[x, y, z, w, l, h, yaw, vx, vy, class, prob], ...]]
            if box[-1] < self.threshold:
                continue

            o = DetectedObject()
            o.existence_probability = box[-1]
            o.classification.append(ObjectClassification(label=self.nuscenes_to_autoware[int(box[-2])], probability=box[-1]))
            o.kinematics.pose_with_covariance.pose.position.x = box[0]
            o.kinematics.pose_with_covariance.pose.position.y = box[1]
            o.kinematics.pose_with_covariance.pose.position.z = box[2]
            o.shape.dimensions.y = box[3]
            o.shape.dimensions.x = box[4]
            o.shape.dimensions.z = box[5]
            o.shape.type = Shape.BOUNDING_BOX
            yaw = box[6] + np.pi/2
            o.kinematics.pose_with_covariance.pose.orientation.z = -np.sin(yaw/2)
            o.kinematics.pose_with_covariance.pose.orientation.w = np.cos(yaw/2)
            o.kinematics.has_position_covariance = False
            o.kinematics.orientation_availability = DetectedObjectKinematics.AVAILABLE
            o.kinematics.has_twist = True
            o.kinematics.has_twist_covariance = False
            vel_angle = np.arctan2(box[7], box[8]) - box[6] + np.pi
            speed = np.sqrt(box[7]**2 + box[8]**2)
            o.kinematics.twist_with_covariance.twist.linear.x = speed * np.cos(vel_angle)
            o.kinematics.twist_with_covariance.twist.linear.y = speed * np.sin(vel_angle)
            objects.objects.append(o)

        self.objects_pub.publish(objects)

    def save_data(self, lidar2image, points):
        """Save data for debugging"""
        data  = "example-data"
        self.get_logger().info(f"Saving data to {data}")
        tensor.save(points, f"{data}/points.tensor")
        tensor.save(self.camera_intrinsics, f"{data}/camera_intrinsics.tensor")
        tensor.save(self.camera2lidar, f"{data}/camera2lidar.tensor")
        tensor.save(self.img_aug_matrix, f"{data}/img_aug_matrix.tensor")
        tensor.save(lidar2image, f"{data}/lidar2image.tensor")
        cv2.imwrite(f"{data}/0-FRONT.jpg", self.images[0, 0][..., ::-1])
        cv2.imwrite(f"{data}/1-FRONT_RIGHT.jpg", self.images[0, 1][..., ::-1])
        cv2.imwrite(f"{data}/2-FRONT_LEFT.jpg", self.images[0, 2][..., ::-1])
        cv2.imwrite(f"{data}/3-BACK.jpg", self.images[0, 3][..., ::-1])
        cv2.imwrite(f"{data}/4-BACK_LEFT.jpg", self.images[0, 4][..., ::-1])
        cv2.imwrite(f"{data}/5-BACK_RIGHT.jpg", self.images[0, 5][..., ::-1])

def main():
    import sys
    rclpy.init(args=sys.argv)

    # create node and spin
    node = BEVFusion()
    executor=rclpy.executors.MultiThreadedExecutor(num_threads=2)
    rclpy.spin(node, executor)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
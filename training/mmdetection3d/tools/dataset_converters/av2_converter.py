import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import mmengine
import math
import copy
import numpy as np
from tqdm import tqdm 

from pathlib import Path
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.datasets.convert_utils import NuScenesNameMapping
from mmdet3d.structures import points_cam2img

from av2.datasets.sensor.sensor_dataloader import read_city_SE3_ego
from av2.datasets.sensor.splits import TRAIN, VAL, TEST
from av2.utils.io import read_feather
from av2.geometry.geometry import quat_to_mat
from av2.geometry.se3 import SE3
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.structures.cuboid import Cuboid, CuboidList
from shapely.geometry import MultiPoint, box
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

class_names = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG')


TRAIN_SAMPLE_RATE = 10
VAL_SAMPLE_RATE = 5
VELOCITY_SAMPLING_RATE = 5

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.
    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.
    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None
def generate_record(x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str, cat_name: str) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.
    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.
    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    coco_rec = dict()

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = class_names.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec

def yaw_to_quaternion3d(yaw: float) -> np.ndarray:
    """Convert a rotation angle in the xy plane (i.e. about the z axis) to a quaternion.
    Args:
        yaw: angle to rotate about the z-axis, representing an Euler angle, in radians
    Returns:
        array w/ quaternion coefficients (qw,qx,qy,qz) in scalar-first order, per Argoverse convention.
    """
    qx, qy, qz, qw = Rotation.from_euler(seq="z", angles=yaw, degrees=False).as_quat()
    return np.array([qw, qx, qy, qz])

def box_velocity(current_annotation, current_timestamp_ns, all_timestamps, annotations, log_dir):
    timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir)
    city_SE3_ego_reference = timestamp_city_SE3_ego_dict[current_timestamp_ns]

    curr_index = all_timestamps.index(current_timestamp_ns)
    prev_index = curr_index - VELOCITY_SAMPLING_RATE
    next_index = curr_index + VELOCITY_SAMPLING_RATE

    track_uuid = current_annotation[1]["track_uuid"]

    if prev_index > 0:
        prev_timestamp_ns = all_timestamps[prev_index]

        #get annotation in prev timestamp
        prev_annotations = annotations[annotations["timestamp_ns"] == int(prev_timestamp_ns)]
        prev_annotation = prev_annotations[prev_annotations["track_uuid"] == track_uuid]

        if len(prev_annotation) == 0:
            prev_annotation = None
    else:
        prev_annotation = None 

    if next_index < len(all_timestamps):
        next_timestamp_ns = all_timestamps[next_index]

        #get annotation in next timestamp
        next_annotations = annotations[annotations["timestamp_ns"] == int(next_timestamp_ns)]
        next_annotation = next_annotations[next_annotations["track_uuid"] == track_uuid]

        if len(next_annotation) == 0:
            next_annotation = None
    else:
        next_annotation = None 

    if prev_annotation is None and next_annotation is None:
        return np.array([0, 0, 0])

    # take centered average of displacement for velocity
    if prev_annotation is not None and next_annotation is not None:
        city_SE3_ego_prev = timestamp_city_SE3_ego_dict[prev_timestamp_ns]
        reference_SE3_ego_prev = city_SE3_ego_reference.inverse().compose(city_SE3_ego_prev)

        city_SE3_ego_next = timestamp_city_SE3_ego_dict[next_timestamp_ns]
        reference_SE3_ego_next = city_SE3_ego_reference.inverse().compose(city_SE3_ego_next)

        prev_translation = np.array([prev_annotation["tx_m"].item(), prev_annotation["ty_m"].item(), prev_annotation["tz_m"].item()])   
        next_translation = np.array([next_annotation["tx_m"].item(), next_annotation["ty_m"].item(), next_annotation["tz_m"].item()])   

        #convert prev and next annotations into the current annotation reference frame
        prev_translation = reference_SE3_ego_prev.transform_from(prev_translation)
        next_translation = reference_SE3_ego_next.transform_from(next_translation)

        delta_t = (next_timestamp_ns - prev_timestamp_ns) * 1e-9
        return (next_translation - prev_translation) / delta_t

    # take one-sided average of displacement for velocity
    else:
        if prev_annotation is not None:
            city_SE3_ego_prev = timestamp_city_SE3_ego_dict[prev_timestamp_ns]
            reference_SE3_ego_prev = city_SE3_ego_reference.inverse().compose(city_SE3_ego_prev)

            prev_translation = np.array([prev_annotation["tx_m"].item(), prev_annotation["ty_m"].item(), prev_annotation["tz_m"].item()])   
            current_translation = np.array([current_annotation[1]["tx_m"], current_annotation[1]["ty_m"], current_annotation[1]["tz_m"]])   

            #convert prev annotation into the current annotation reference frame
            prev_translation = reference_SE3_ego_prev.transform_from(prev_translation)

            delta_t = (current_timestamp_ns - prev_timestamp_ns) * 1e-9
            return (current_translation - prev_translation) / delta_t

        if next_annotation is not None:
            city_SE3_ego_next = timestamp_city_SE3_ego_dict[next_timestamp_ns]
            reference_SE3_ego_next = city_SE3_ego_reference.inverse().compose(city_SE3_ego_next)

            current_translation = np.array([current_annotation[1]["tx_m"], current_annotation[1]["ty_m"], current_annotation[1]["tz_m"]])   
            next_translation = np.array([next_annotation["tx_m"].item(), next_annotation["ty_m"].item(), next_annotation["tz_m"].item()])   

            #convert next annotations into the current annotation reference frame
            next_translation = reference_SE3_ego_next.transform_from(next_translation)
            
            delta_t = (next_timestamp_ns - current_timestamp_ns) * 1e-9
            return (next_translation - current_translation) / delta_t

def aggregate_sweeps(log_dir, timestamp_ns, num_sweeps = 5):
    lidar_dir = log_dir / "sensors" / "lidar"
    sweep_paths = sorted(lidar_dir.glob("*.feather"))
    timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir)
    city_SE3_ego_reference = timestamp_city_SE3_ego_dict[timestamp_ns]

    reference_index = sweep_paths.index(lidar_dir / f"{timestamp_ns}.feather")
    start = max(0, reference_index - num_sweeps + 1)
    end = reference_index + 1
    sweeps_list = []
    transform_list = []
    delta_list = []

    for i in range(start, end):
        timestamp_ns_i = int(sweep_paths[i].stem)

        sweeps_list.append(sweep_paths[i])
        timestamp_delta = abs(timestamp_ns_i - timestamp_ns)

        delta_list.append(timestamp_delta)
        assert timestamp_ns >= timestamp_ns_i
        if timestamp_delta != 0:
            city_SE3_ego_ti = timestamp_city_SE3_ego_dict[timestamp_ns_i]
            reference_SE3_ego_ti = city_SE3_ego_reference.inverse().compose(city_SE3_ego_ti)
            transform_list.append(reference_SE3_ego_ti)
        else:
            city_SE3_ego_t = timestamp_city_SE3_ego_dict[timestamp_ns]
            reference_SE3_ego_t = city_SE3_ego_reference.inverse().compose(city_SE3_ego_t)
            transform_list.append(reference_SE3_ego_t)
    
    while len(sweeps_list) < num_sweeps:
        sweeps_list.append(sweeps_list[-1])
        transform_list.append(transform_list[-1])
        delta_list.append(delta_list[-1])

    sweeps_list = sweeps_list[::-1]
    transform_list = transform_list[::-1]
    delta_list = delta_list[::-1]

    return sweeps_list, transform_list, delta_list

def sweeps_to_dict(sweeps):
    for sweep in sweeps:
        sweep = {'lidar_points': {'lidar_path': sweep, 'num_pts_feats': 5}}
    return sweeps

def generate_info(filename, log_id, log_dir, annotations, name2cid, all_timestamps, n_sweep):
    timestamp_ns = int(filename.split(".")[0])
    lidar_path = "{log_dir}/sensors/lidar/{timestamp_ns}.feather".format(log_dir=log_dir,timestamp_ns=timestamp_ns)

    # mmcv.check_file_exist(lidar_path)
    mmengine.is_filepath(lidar_path)
    
    if annotations is None:
        gt_bboxes_3d = []
        gt_labels = []
        gt_names = [] 
        gt_num_pts = []
        gt_velocity = []
        gt_uuid = []
        
    else:
        curr_annotations = annotations[annotations["timestamp_ns"] == timestamp_ns]
        curr_annotations = curr_annotations[curr_annotations["num_interior_pts"] > 0]

        gt_bboxes_3d = []
        gt_labels = []
        gt_names = [] 
        gt_num_pts = []
        gt_velocity = []
        gt_uuid = []

        for annotation in curr_annotations.iterrows():
            class_name = annotation[1]["category"]

            if class_name not in class_names:
                continue 

            track_uuid = annotation[1]["track_uuid"]
            num_interior_pts = annotation[1]["num_interior_pts"]

            gt_labels.append(name2cid[class_name])
            gt_names.append(class_name)
            gt_num_pts.append(num_interior_pts)
            gt_uuid.append(track_uuid)

            translation = np.array([annotation[1]["tx_m"], annotation[1]["ty_m"], annotation[1]["tz_m"]])
            lwh = np.array([annotation[1]["length_m"], annotation[1]["width_m"], annotation[1]["height_m"]])
            rotation = quat_to_mat(np.array([annotation[1]["qw"], annotation[1]["qx"], annotation[1]["qy"], annotation[1]["qz"]]))
            ego_SE3_object = SE3(rotation=rotation, translation=translation)

            rot = ego_SE3_object.rotation
            lwh = lwh.tolist()
            center = translation.tolist()
            center[2] = center[2] - lwh[2] / 2
            yaw = math.atan2(rot[1, 0], rot[0, 0])

            gt_bboxes_3d.append([*center, *lwh, yaw])

            velocity = box_velocity(annotation,timestamp_ns, all_timestamps, annotations, Path(log_dir))[:2]
            gt_velocity.append(velocity)

    sweeps, transforms, deltas = aggregate_sweeps(Path(log_dir), timestamp_ns, n_sweep)
    sweeps = [{'lidar_points': {'lidar_path': str(sweep), 'num_pts_feats': 5}} for sweep in sweeps]
    
    info = {
        'lidar_points':{'lidar_path': lidar_path,
                        'num_pts_feats': 5,
                        },
        'log_id': log_id,
        'lidar_sweeps' : sweeps, 
        'transforms' : transforms,
        'timestamp': timestamp_ns,
        'timestamp_deltas' :deltas,
        'gt_bboxes' : gt_bboxes_3d,
        'gt_labels' : gt_labels,
        'gt_names' : gt_names, 
        'gt_num_pts' : gt_num_pts,
        'gt_velocity' : gt_velocity,
        'gt_uuid' : gt_uuid,
    }

    return info 


def create_av2_infos(root_path, info_prefix, out_dir, max_sweeps=5):
    os.makedirs(out_dir, exist_ok=True)

    name2cid = {c: i for i, c in enumerate(class_names)}
    n_sweep = max_sweeps

    train_infos = []
    val_infos = []
    test_infos = []

    # for log_id in tqdm(TRAIN):
    #     split = "train"
    #     log_dir = "{root_path}/{split}/{log_id}".format(root_path=root_path, split=split, log_id=log_id)
    #     lidar_paths = "{log_dir}/sensors/lidar".format(log_dir=log_dir)
    #     annotations_path = "{log_dir}/annotations.feather".format(log_dir=log_dir)
    #     annotations = read_feather(Path(annotations_path))
    #     mmengine.is_filepath(annotations_path)
    #     all_timestamps = sorted([int(filename.split(".")[0]) for filename in os.listdir(lidar_paths)])
    #     for i, filename in enumerate(sorted(os.listdir(lidar_paths))):
    #         if i % TRAIN_SAMPLE_RATE != 0:
    #             continue 
    #         info = generate_info(filename, log_id, log_dir, annotations, name2cid, all_timestamps, n_sweep)
    #         train_infos.append(info)
    # out_file = "{out_dir}/{info_prefix}_infos_{split}.pkl".format(out_dir=out_dir,info_prefix=info_prefix, split=split)
    # train_infos_v2 = update_av2_infos(train_infos, out_dir)
    # metainfo = dict()
    # metainfo['categories'] = {k: i for i, k in enumerate(class_names)}
    # metainfo['dataset'] = 'av2dataset'
    # train_data_infos = dict(metainfo=dict(), data_list=train_infos_v2)
    # mmengine.dump(train_data_infos, out_file)
    for log_id in tqdm(VAL):
        split = "val"
        log_dir = "{root_path}/{split}/{log_id}".format(root_path=root_path, split=split, log_id=log_id)
        lidar_paths = "{log_dir}/sensors/lidar".format(log_dir=log_dir)
        annotations_path = "{log_dir}/annotations.feather".format(log_dir=log_dir)
        annotations = read_feather(Path(annotations_path))
        mmengine.is_filepath(annotations_path)
        all_timestamps = sorted([int(filename.split(".")[0]) for filename in os.listdir(lidar_paths)])
        for i, filename in enumerate(sorted(os.listdir(lidar_paths))):
            if i % VAL_SAMPLE_RATE != 0:
                continue 
            info = generate_info(filename, log_id, log_dir, annotations, name2cid, all_timestamps, n_sweep)
            val_infos.append(info)
    out_file = "{out_dir}/{info_prefix}_infos_{split}.pkl".format(out_dir=out_dir,info_prefix=info_prefix, split=split)
    val_infos_v2 = update_av2_infos(val_infos, out_dir)
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(class_names)}
    metainfo['dataset'] = 'av2dataset'
    val_data_infos = dict(metainfo=metainfo, data_list=val_infos_v2)
    mmengine.dump(val_data_infos, out_file)
    
    for log_id in tqdm(TEST):
        split = "test"
        log_dir = "{root_path}/{split}/{log_id}".format(root_path=root_path, split=split, log_id=log_id)
        lidar_paths = "{log_dir}/sensors/lidar".format(log_dir=log_dir)
        annotations = None
        all_timestamps = sorted([int(filename.split(".")[0]) for filename in os.listdir(lidar_paths)])
        for i, filename in enumerate(sorted(os.listdir(lidar_paths))):
            if i % VAL_SAMPLE_RATE != 0:
                continue 
            info = generate_info(filename, log_id, log_dir, annotations, name2cid, all_timestamps, n_sweep)
            test_infos.append(info)
    out_file = "{out_dir}/{info_prefix}_infos_{split}.pkl".format(out_dir=out_dir,info_prefix=info_prefix, split=split)
    mmengine.dump(test_infos, out_file)

def update_av2_infos(data_list, out_dir):
    converted_list = []
    print("Updating the info files to v2 format...")
    for i, ori_info_dict in enumerate(
        mmengine.track_iter_progress(data_list)):
        temp_data_info = get_empty_standard_data_info()
        temp_data_info['sample_idx'] = i
        temp_data_info['log_id'] = ori_info_dict['log_id']
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict.get(
            'num_features', 5)
        temp_data_info['lidar_points']['lidar_path'] = ori_info_dict['lidar_points']['lidar_path']
        for ori_sweep in ori_info_dict['lidar_sweeps']:
            temp_lidar_sweep = get_single_lidar_sweep()
            temp_lidar_sweep['lidar_points']['lidar_path'] = ori_sweep['lidar_points']['lidar_path']
            temp_data_info['lidar_sweeps'].append(temp_lidar_sweep)
        ignore_class_name = set()
        if 'gt_bboxes' in ori_info_dict:
            num_instances = len(ori_info_dict['gt_bboxes'])
            print('NUM',num_instances)
            for i in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox_3d'] = ori_info_dict['gt_bboxes'][i]
                if ori_info_dict['gt_names'][i] in class_names:
                    empty_instance['bbox_label'] = class_names.index(
                        ori_info_dict['gt_names'][i])
                else:
                    ignore_class_name.add(ori_info_dict['gt_names'][i])
                    empty_instance['bbox_label'] = -1
                empty_instance['bbox_label_3d'] = copy.deepcopy(
                    empty_instance['bbox_label'])
                empty_instance['velocity'] = ori_info_dict['gt_velocity'][i]
                    # i, :].tolist()
                empty_instance['num_lidar_pts'] = ori_info_dict[
                    'gt_num_pts'][i]
                empty_instance = clear_instance_unused_keys(empty_instance)
                temp_data_info['instances'].append(empty_instance)
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    return converted_list
    
def get_2d_boxes(info, cam_model, width, height, cam_img, mono3d=True):
    timestamp = info["timestamp"]
    log_id = info["log_id"]
    ego_SE3_cam = cam_model.ego_SE3_cam
    camera_intrinsic = cam_model.intrinsics.K
    
    coco_infos = []
    for name, bbox, velocity, num_pts, uuid in zip(info["gt_names"], info["gt_bboxes"], info["gt_velocity"], info["gt_num_pts"], info["gt_uuid"]):
        quat = yaw_to_quaternion3d(bbox[-1]).tolist()
        cuboid_ego = Cuboid.from_numpy(np.array(bbox[:-1] + quat), name, timestamp)
        
        cuboid_ego_av2 = copy.deepcopy(cuboid_ego)
        cuboid_ego_av2.dst_SE3_object.translation[2] = cuboid_ego_av2.dst_SE3_object.translation[2] + cuboid_ego_av2.height_m / 2
        cuboid_cam = cuboid_ego_av2.transform(ego_SE3_cam.inverse())
        
        cam_box = CuboidList([cuboid_cam])
        cuboids_vertices_cam = cam_box.vertices_m
        N, V, D = cuboids_vertices_cam.shape

        # Collapse first dimension to allow for vectorization.
        cuboids_vertices_cam = cuboids_vertices_cam.reshape(-1, D)
        _, _, is_valid = cam_model.project_cam_to_img(cuboids_vertices_cam)

        num_valid = np.sum(is_valid)
        if num_valid > 0:
            corner_coords = view_points(cuboid_cam.vertices_m.T, camera_intrinsic, True).T[:, :2].tolist()

            # Keep only corners that fall within the image.
            final_coords = post_process_coords(corner_coords, (width, height))

            # Skip if the convex hull of the re-projected corners
            # does not intersect the image canvas.
            if final_coords is None:
                continue
            else:
                min_x, min_y, max_x, max_y = final_coords

            repro_rec = generate_record(min_x, min_y, max_x, max_y, log_id, cam_img, name)

            if mono3d and (repro_rec is not None):
                rot = cuboid_ego.dst_SE3_object.rotation
                size = [cuboid_ego.length_m, cuboid_ego.width_m, cuboid_ego.height_m]
                center = cuboid_ego.dst_SE3_object.translation.tolist()
                yaw = math.atan2(rot[1, 0], rot[0, 0]) - cam_model.egovehicle_yaw_cam_rad

                repro_rec['bbox_cam3d'] = [*center, *size, yaw]
                repro_rec['velo_cam3d'] = ego_SE3_cam.transform_from([*velocity, 0])[:2]

                center2d = points_cam2img(cuboid_cam.dst_SE3_object.translation.tolist(), camera_intrinsic, with_depth=True)

                repro_rec['center2d'] = center2d.squeeze().tolist()
                # normalized center2D + depth
                # if samples with depth < 0 will be removed
                if repro_rec['center2d'][2] <= 0:
                    continue

            repro_rec['attribute_name'] = "None"
            repro_rec['attribute_id'] = 0
            repro_rec['gt_num_pts'] = num_pts
            repro_rec['gt_uuid'] = uuid

            coco_infos.append(repro_rec)

    return coco_infos

def export_2d_annotation(root_path, info_path, mono3d=True):
    """Export 2d annotation from the info file and raw data.
    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'ring_front_center',
        'ring_front_left',
        'ring_front_right',
        'ring_side_left',
        'ring_side_right',
        'ring_rear_left',
        'ring_rear_right',
    ]
    av2_infos = mmengine.load(info_path)
    cat2Ids = [
        dict(id=class_names.index(cat_name), name=cat_name)
        for cat_name in class_names
    ]

    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)

    for info in mmengine.utils.track_iter_progress(av2_infos):
        log_id = info["log_id"]
        timestamp = info["timestamp"]

        log_dir = Path("{}/{}".format(root_path, log_id))
        
        cam_imgs, cam_models = {}, {}
        for cam_name in camera_types:
            cam_models[cam_name] = PinholeCamera.from_feather(log_dir, cam_name)

            cam_path = root_path + "/{}/sensors/cameras/{}/".format(log_id, cam_name)
            closest_dst = np.inf
            closest_img = None
            for filename in os.listdir(cam_path):
                img_timestamp = int(filename.split(".")[0])
                delta = abs(timestamp - img_timestamp)

                if delta < closest_dst:
                    closest_img = cam_path + filename
                    closest_dst = delta
                    
            cam_imgs[cam_name] = closest_img

        for cam_name in camera_types:
            cam_img = mmcv.imread(cam_imgs[cam_name])
            (height, width, _) = cam_img.shape

            coco_infos = get_2d_boxes(info, copy.deepcopy(cam_models[cam_name]), width, height, cam_imgs[cam_name], mono3d=True)
            
            coco_2d_dict['images'].append(
                dict(
                file_name=cam_imgs[cam_name],
                timestamp=timestamp,
                id=cam_name,
                token=log_id,
                ego_SE3_cam_rotation=cam_models[cam_name].ego_SE3_cam.rotation,
                ego_SE3_cam_translation=cam_models[cam_name].ego_SE3_cam.translation,
                ego_SE3_cam_intrinsics=cam_models[cam_name].intrinsics.K,
                width=width,
                height=height))

            coco_2d_dict['annotations'].append(coco_infos)
                    
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'

    mmengine.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_empty_standard_data_info(
        camera_types=['CAM0', 'CAM1', 'CAM2', 'CAM3', 'CAM4']):

    data_info = dict(
        # (str): Sample id of the frame.
        sample_idx=None,
        # (str, optional): '000010'
        token=None,
        **get_single_image_sweep(camera_types),
        # (dict, optional): dict contains information
        # of LiDAR point cloud frame.
        lidar_points=get_empty_lidar_points(),
        # (dict, optional) Each dict contains
        # information of Radar point cloud frame.
        # (list[dict], optional): Image sweeps data.
        image_sweeps=[],
        lidar_sweeps=[],
        instances=[],
        # (list[dict], optional): Required by object
        # detection, instance  to be ignored during training.
        instances_ignore=[],
        # (str, optional): Path of semantic labels for each point.
        pts_semantic_mask_path=None,
        # (str, optional): Path of instance labels for each point.
        pts_instance_mask_path=None)
    return data_info

def clear_instance_unused_keys(instance):
    keys = list(instance.keys())
    for k in keys:
        if instance[k] is None:
            del instance[k]
    return instance


def get_single_lidar_sweep():
    single_lidar_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None,
        # (dict): Information of images captured by multiple cameras
        lidar_points=get_empty_lidar_points())
    return single_lidar_sweep


def get_single_image_sweep(camera_types):
    single_image_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None)
    # (dict): Information of images captured by multiple cameras
    images = dict()
    for cam_type in camera_types:
        images[cam_type] = get_empty_img_info()
    single_image_sweep['images'] = images
    return single_image_sweep

def get_empty_img_info():
    img_info = dict(
        # (str, required): the path to the image file.
        img_path=None,
        # (int) The height of the image.
        height=None,
        # (int) The width of the image.
        width=None,
        # (str, optional): Path of the depth map file
        depth_map=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to image with
        # shape [3, 3], [3, 4] or [4, 4].
        cam2img=None,
        # (list[list[float]]): Transformation matrix from lidar
        # or depth to image with shape [4, 4].
        lidar2img=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to ego-vehicle
        # with shape [4, 4].
        cam2ego=None)
    return img_info

def get_empty_lidar_points():
    lidar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of LiDAR data file.
        lidar_path=None,
        # (list[list[float]], optional): Transformation matrix
        # from lidar to ego-vehicle
        # with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        lidar2ego=None,
    )
    return lidar_points

def get_empty_instance():
    """Empty annotation for single instance."""
    instance = dict(
        # (list[float], required): list of 4 numbers representing
        # the bounding box of the instance, in (x1, y1, x2, y2) order.
        bbox=None,
        # (int, required): an integer in the range
        # [0, num_categories-1] representing the category label.
        bbox_label=None,
        #  (list[float], optional): list of 7 (or 9) numbers representing
        #  the 3D bounding box of the instance,
        #  in [x, y, z, w, h, l, yaw]
        #  (or [x, y, z, w, h, l, yaw, vx, vy]) order.
        bbox_3d=None,
        # (bool, optional): Whether to use the
        # 3D bounding box during training.
        bbox_3d_isvalid=None,
        # (int, optional): 3D category label
        # (typically the same as label).
        bbox_label_3d=None,
        # (float, optional): Projected center depth of the
        # 3D bounding box compared to the image plane.
        depth=None,
        #  (list[float], optional): Projected
        #  2D center of the 3D bounding box.
        center_2d=None,
        # (int, optional): Attribute labels
        # (fine-grained labels such as stopping, moving, ignore, crowd).
        attr_label=None,
        # (int, optional): The number of LiDAR
        # points in the 3D bounding box.
        num_lidar_pts=None,
        # (int, optional): The number of Radar
        # points in the 3D bounding box.
        num_radar_pts=None,
        # (int, optional): Difficulty level of
        # detecting the 3D bounding box.
        difficulty=None,
        unaligned_bbox_3d=None)
    return instance

def clear_data_info_unused_keys(data_info):
    keys = list(data_info.keys())
    empty_flag = True
    for key in keys:
        # we allow no annotations in datainfo
        if key in ['instances', 'cam_sync_instances', 'cam_instances']:
            empty_flag = False
            continue
        if isinstance(data_info[key], list):
            if len(data_info[key]) == 0:
                del data_info[key]
            else:
                empty_flag = False
        elif data_info[key] is None:
            del data_info[key]
        elif isinstance(data_info[key], dict):
            _, sub_empty_flag = clear_data_info_unused_keys(data_info[key])
            if sub_empty_flag is False:
                empty_flag = False
            else:
                # sub field is empty
                del data_info[key]
        else:
            empty_flag = False

    return data_info, empty_flag
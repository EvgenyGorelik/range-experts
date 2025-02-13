# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Callable, List, Union
import warnings
import mmcv
import mmengine
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
from mmdet3d.structures import get_box_type
from mmengine.dataset import Compose
from .det3d_dataset import Det3DDataset
from .utils import get_loading_pipeline
from scipy.spatial.transform import Rotation
from pathlib import Path
from collections import defaultdict
import os 

def yaw_to_quaternion3d(yaw: float) -> np.ndarray:
    """Convert a rotation angle in the xy plane (i.e. about the z axis) to a quaternion.
    Args:
        yaw: angle to rotate about the z-axis, representing an Euler angle, in radians
    Returns:
        array w/ quaternion coefficients (qw,qx,qy,qz) in scalar-first order, per Argoverse convention.
    """
    qx, qy, qz, qw = Rotation.from_euler(seq="z", angles=yaw, degrees=False).as_quat()
    return np.array([qw, qx, qy, qz])

def distance_matrix(A, B, squared=False):
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared

def extract_result_dict(results, key):
    """Extract and return the data corresponding to key in result dict.

    ``results`` is a dict output from `pipeline(input_dict)`, which is the
        loaded data from ``Dataset`` class.
    The data terms inside may be wrapped in list, tuple and DataContainer, so
        this function essentially extracts data from these wrappers.

    Args:
        results (dict): Data loaded using pipeline.
        key (str): Key of the desired data.

    Returns:
        np.ndarray | torch.Tensor: Data term.
    """
    if key not in results.keys():
        return None
    # results[key] may be data or list[data] or tuple[data]
    # data may be wrapped inside DataContainer
    data = results[key]
    if isinstance(data, (list, tuple)):
        data = data[0]
    if isinstance(data, mmcv.parallel.DataContainer):
        data = data._data
    return data

@DATASETS.register_module()
class AV2Dataset(Det3DDataset):   
    CLASSES = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG')
    METAINFO = {
        'classes':('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG'),
    'palette': [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
                (0, 255, 255), (255, 0, 255), (255, 255, 255), (128, 0, 0), (0, 128, 0),
                (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (192, 192, 192), (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128),
                (0, 128, 128), (0, 0, 128), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
                
    }
    def __init__(self,
                data_root,
                ann_file,
                pipeline=None,
                classes=None,
                modality=dict(
                     use_camera=False,
                     use_lidar=True,
                 ),
                data_prefix = dict(pts='', img=''),
                box_type_3d='LiDAR',
                filter_empty_gt=True,
                test_mode=False,
                with_velocity: bool = True,
                file_client_args=dict(backend='disk'), 
                sampler_type=None,
                task_names=None,
                class_mapping=None,
                use_valid_flag=False,
                **kwargs):
        self.data_root = data_root
        self.ann_file = ann_file
        self.with_velocity = with_velocity
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.sampler_type=sampler_type
        self.task_names=task_names
        self.class_mapping=class_mapping
        
        self.CLASSES = self.get_classes(classes)
        # self.file_client = mmcv.FileClient(**file_client_args)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo = self.METAINFO,
            data_prefix=data_prefix,
            modality=modality,
            pipeline=pipeline,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        if not self.test_mode:
            self._set_group_flag()
    
    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.
        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        return mmengine.load(ann_file, file_format='pkl')

    def parse_ann_info(self, info: dict):
        ann_info = super().parse_ann_info(info)
        if ann_info is not None:
            if self.with_velocity:
                gt_bboxes_3d = ann_info['gt_bboxes_3d']
                gt_velocities = ann_info['velocities']
                nan_mask = np.isnan(gt_velocities[:, 0])
                gt_velocities[nan_mask] = [0.0, 0.0]
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocities],
                                              axis=-1)
                ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        else:
            # empty instance
            ann_info = dict()
            if self.with_velocity:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            else:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_bboxes_labels'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['attr_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)
        return ann_info

    def parse_data_info(self, info:dict):
        if self.modality['use_lidar']:
            info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']
        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    if cam_id in self.data_prefix:
                        cam_prefix = self.data_prefix[cam_id]
                    else:
                        cam_prefix = self.data_prefix.get('img', '')
                img_info['img_path'] = osp.join(cam_prefix,
                                                img_info['img_path'])
            if not self.test_mode:
            # used in training
                info['ann_info'] = self.parse_ann_info(info)
            if self.test_mode and self.load_eval_anns:
                info['eval_ann_info'] = self.parse_ann_info(info)
        return info

    # def parse_data_info(self, index):
    #     """Get data info according to the given index.
    #     Args:
    #         index (int): Index of the sample data to get.
    #     Returns:
    #         dict: Data information that will be passed to the data
    #             preprocessing pipelines. It includes the following keys:
    #             - sample_idx (str): Sample index.
    #             - pts_filename (str): Filename of point clouds.
    #             - file_name (str): Filename of point clouds.
    #             - ann_info (dict): Annotation info.
    #     """
    #     info = self.data_infos[index]
    #     sample_idx = info['log_id']
  
    #     input_dict = dict(
    #         lidar_path=info['lidar_path'],
    #         sample_idx=sample_idx,
    #         sweeps=info['sweeps'],
    #         transforms=info['transforms'],
    #         timestamp_deltas=info['timestamp_deltas'],
    #         timestamp=info['timestamp'])

    #     if not self.test_mode:
    #         annos = self.get_ann_info(index)
    #         input_dict['ann_info'] = annos
    #         if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
    #             return None
    #     return input_dict
    
    def get_ann_info(self, index):
        """Get annotation info according to the given index.
        Args:
            index (int): Index of the annotation data to get.
        Returns:
            dict: Annotation information consists of the following keys:
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        gt_bboxes_3d = info['gt_bboxes']
        gt_names_3d = info['gt_names']
        gt_labels_3d = np.array(info['gt_labels'])

        gt_velocity = info['gt_velocity']
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1])

        if "gt_uuid" in info:
            uuid = info['gt_uuid']
        else:
            uuid = [None] * len(gt_bboxes_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_uuid = uuid)

        return anns_results

    def pre_pipeline(self, results):
        """Initialization before data preparation.
        Args:
            results (dict): Dict before data preprocessing.
                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example
    
    def prepare_test_data(self, index):
        """Prepare data for testing.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
    
    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.
        Args:
            classes (Sequence[str] | str): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names
    
    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        raise NotImplementedError('_build_default_pipeline is not implemented '
                                  f'for dataset {self.__class__.__name__}')

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.
        Args:
            pipeline (list[dict]): Input pipeline. If None is given,
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn(
                    'Use default pipeline for data loading, this may cause '
                    'errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)
    
    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.
        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.
        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data
    
    # def __len__(self):
    #     """Return the length of data infos.
    #     Returns:
    #         int: Length of data infos.
    #     """
    #     return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.
        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
    
    def format_results(self, outputs):
        """Format the results to be recognizable to the Argoverse eval script.
        Args:
            outputs (list[dict]): Testing results of the dataset.
            dt_root (str):
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """

        predictions = {"tx_m" : [],
                       "ty_m" : [],
                       "tz_m" : [],
                       "length_m" : [],
                       "width_m" : [],
                       "height_m" : [],
                       "qw" : [],
                       "qx" : [],
                       "qy" : [],
                       "qz" : [],
                       "vx" : [],
                       "vy" : [],
                       "score" : [],
                       "log_id" : [],
                       "timestamp_ns" : [],
                       "category" : []}

        ground_truth = {"tx_m" : [],
                       "ty_m" : [],
                       "tz_m" : [],
                       "length_m" : [],
                       "width_m" : [],
                       "height_m" : [],
                       "qw" : [],
                       "qx" : [],
                       "qy" : [],
                       "qz" : [],
                       "vx" : [],
                       "vy" : [],
                       "score" : [],
                       "num_interior_pts" : [],
                       "log_id" : [],
                       "track_uuid" : [],
                       "timestamp_ns" : [],
                       "category" : []}

        for index, pred_dict in enumerate(mmcv.track_iter_progress(outputs)):
            info = self.data_infos[index]
            log_id = info["log_id"]
            
            timestamp_ns = info["timestamp"]

            if "gt_uuid" in info:
                track_uuid = info['gt_uuid']
            else:
                track_uuid = [None] * len(info['gt_bboxes'])

            for bbox, label, velocity, num_pts, uuid in zip(
                info['gt_bboxes'],
                info['gt_labels'],
                info['gt_velocity'],
                info['gt_num_pts'],
                track_uuid
            ):
                quat = yaw_to_quaternion3d(bbox[6]).tolist()
        
                ground_truth['tx_m'].append(bbox[0])
                ground_truth['ty_m'].append(bbox[1])
                ground_truth['tz_m'].append(bbox[2] + bbox[5] / 2)

                ground_truth['qw'].append(quat[0])
                ground_truth['qx'].append(quat[1])
                ground_truth['qy'].append(quat[2])
                ground_truth['qz'].append(quat[3])

                ground_truth['length_m'].append(bbox[3])
                ground_truth['width_m'].append(bbox[4])
                ground_truth['height_m'].append(bbox[5])

                ground_truth['vx'].append(velocity[0])
                ground_truth['vy'].append(velocity[1])

                ground_truth['category'].append(self.CLASSES[label])
                ground_truth['score'].append(-1)
                ground_truth['num_interior_pts'].append(num_pts)
                ground_truth['track_uuid'].append(uuid)

                ground_truth['log_id'].append(log_id)
                ground_truth['timestamp_ns'].append(timestamp_ns)
            
            for bbox, score, label in zip(
                pred_dict['pts_bbox']['boxes_3d'],
                pred_dict['pts_bbox']['scores_3d'],
                pred_dict['pts_bbox']['labels_3d'],
            ):
                bbox = bbox.numpy()
                quat = yaw_to_quaternion3d(bbox[6]).tolist()

                predictions['tx_m'].append(bbox[0])
                predictions['ty_m'].append(bbox[1])
                predictions['tz_m'].append(bbox[2] + bbox[5] / 2)

                predictions['qw'].append(quat[0])
                predictions['qx'].append(quat[1])
                predictions['qy'].append(quat[2])
                predictions['qz'].append(quat[3])

                predictions['length_m'].append(bbox[3])
                predictions['width_m'].append(bbox[4])
                predictions['height_m'].append(bbox[5])

                predictions['vx'].append(bbox[-2])
                predictions['vy'].append(bbox[-1])

                class_name = self.CLASSES[label.item()] if label.item() < len(self.CLASSES) else "OTHER"
                predictions['category'].append(class_name)
                predictions['score'].append(score.item())

                predictions['log_id'].append(log_id)
                predictions['timestamp_ns'].append(timestamp_ns)

        predictionsDataFrame = pd.DataFrame.from_dict(predictions)
        groundTruthDataFrame = pd.DataFrame.from_dict(ground_truth)

        return predictionsDataFrame, groundTruthDataFrame
    
    def multimodal_filter(self, predictionsDataFrame, rgbPredictionsDataFrame):
        predictions = {"tx_m" : [],
                       "ty_m" : [],
                       "tz_m" : [],
                       "length_m" : [],
                       "width_m" : [],
                       "height_m" : [],
                       "qw" : [],
                       "qx" : [],
                       "qy" : [],
                       "qz" : [],
                       "vx" : [],
                       "vy" : [],
                       "score" : [],
                       "log_id" : [],
                       "timestamp_ns" : [],
                       "category" : []}
        dist_th = 12
        filter_classes = ['BICYCLIST', 'WHEELED_RIDER', 'CONSTRUCTION_CONE', 'SIGN', 'STOP_SIGN', 'BUS', 'TRUCK',  'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS', 'BICYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'DOG']
        sweeps = set()

        lidar_preds, rgb_preds = defaultdict(lambda : defaultdict(dict)), defaultdict(lambda : defaultdict(dict))
        for _, tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz, vx, vy, score, log_id, timestamp_ns, category in predictionsDataFrame.itertuples():
            lidar_preds[log_id][timestamp_ns][category] = []
            sweeps.add((log_id, timestamp_ns, category))

        for _, _, tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz, vx, vy, score, log_id, timestamp_ns, category in rgbPredictionsDataFrame.itertuples():
            rgb_preds[log_id][timestamp_ns][category] = []
            sweeps.add((log_id, timestamp_ns, category))

        for _, tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz, vx, vy, score, log_id, timestamp_ns, category in predictionsDataFrame.itertuples():
            pred = {"tx_m" : tx_m, "ty_m" : ty_m, "tz_m" : tz_m, 
                    "qw" : qw, "qx" : qx, "qy" : qy, "qz" : qz, 
                    "length_m" : length_m, "width_m" : width_m, "height_m" : height_m,
                    "vx" : vx, "vy" : vy,
                    "category" : category, "score" : score, 
                    "log_id" : log_id, "timestamp_ns" : timestamp_ns}

            lidar_preds[log_id][timestamp_ns][category].append(pred)

        for _, _, tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz, vx, vy, score, log_id, timestamp_ns, category in rgbPredictionsDataFrame.itertuples():
            pred = {"tx_m" : tx_m, "ty_m" : ty_m, "tz_m" : tz_m, 
                    "qw" : qw, "qx" : qx, "qy" : qy, "qz" : qz, 
                    "length_m" : length_m, "width_m" : width_m, "height_m" : height_m,
                    "vx" : vx, "vy" : vy,
                    "category" : category, "score" : score, 
                    "log_id" : log_id, "timestamp_ns" : timestamp_ns}

            rgb_preds[log_id][timestamp_ns][category].append(pred)
        
        for log_id, timestamp_ns, class_name in mmcv.track_iter_progress(sweeps):
            if class_name not in lidar_preds[log_id][timestamp_ns]:
                lidar = []
            else:
                lidar = lidar_preds[log_id][timestamp_ns][class_name]   

            lidar_center = np.array([[l["tx_m"], l["ty_m"], l["tz_m"]] for l in lidar])

            if class_name in filter_classes:
                if class_name not in rgb_preds[log_id][timestamp_ns]:
                    rgb = []
                else:
                    rgb = rgb_preds[log_id][timestamp_ns][class_name]  

                rgb_center = np.array([[r["tx_m"], r["ty_m"], r["tz_m"]] for r in rgb])
                
                if len(lidar_center) == 0 or len(rgb_center) == 0:
                    continue 

                dist = np.min(distance_matrix(rgb_center, lidar_center), axis=0) < dist_th
                filtered = np.array(lidar)[dist]
                
            else:
                filtered = lidar

            for pred in filtered:
                predictions['tx_m'].append(pred['tx_m'])
                predictions['ty_m'].append(pred['ty_m'])
                predictions['tz_m'].append(pred['tz_m'])

                predictions['qw'].append(pred['qw'])
                predictions['qx'].append(pred['qx'])
                predictions['qy'].append(pred['qy'])
                predictions['qz'].append(pred['qz'])

                predictions['length_m'].append(pred['length_m'])
                predictions['width_m'].append(pred['width_m'])
                predictions['height_m'].append(pred['height_m'])

                predictions['vx'].append(pred['vx'])
                predictions['vy'].append(pred['vy'])

                predictions['category'].append(pred['category'])
                predictions['score'].append(pred['score'])

                predictions['log_id'].append(pred['log_id'])
                predictions['timestamp_ns'].append(pred['timestamp_ns'])

        predictionsDataFrame = pd.DataFrame.from_dict(predictions)
        return predictionsDataFrame
    
    def evaluate(self, results, out_path=None, **kwargs):
        """Evaluate.
        """
        from av2.evaluation.detection.eval import evaluate, evaluate_hierarchy
        from av2.evaluation.detection.utils import DetectionCfg
        split = "val"
        max_range = 50
        #if out_path is not None:
        #    pipeline = kwargs.get("pipeline", None)
        #    self.show(results, out_path + "/visuals/", show=False, pipeline=pipeline)

        metric_type = kwargs.get("metric_type", None)
        filter = kwargs.get("filter", None)
        predictions = kwargs.get("predictions", None)
        ground_truth = kwargs.get("ground_truth", None)

        if predictions is not None:
            predictionsDataFrame = pd.read_csv(predictions)
            groundTruthDataFrame = pd.read_csv(ground_truth)
            
        else:
            predictionsDataFrame, groundTruthDataFrame = self.format_results(results)

            if out_path is not None:
                pd.DataFrame.to_csv(groundTruthDataFrame, out_path + "/{}_gt.csv".format(split))
                pd.DataFrame.to_csv(predictionsDataFrame, out_path + "/{}_detections.csv".format(split))


            if filter is not None: 
                rgbPredictionsDataFrame = pd.read_csv(filter)
                predictionsDataFrame = self.multimodal_filter(predictionsDataFrame, rgbPredictionsDataFrame)

        cfg = DetectionCfg(dataset_dir = Path("{}/{}".format(self.data_root, split)), max_range_m=max_range)
        
        if metric_type == "standard":
            _, _, metrics = evaluate(predictionsDataFrame, groundTruthDataFrame, cfg)
        
        elif metric_type == "hierarchy":
            metrics = evaluate_hierarchy(predictionsDataFrame, groundTruthDataFrame, cfg)

        else:
            assert False, "Unsupported metric_type {}".format(metric_type)
        print(metrics)
        if out_path is not None:
            filter_tag = "_filter" if filter is not None else ""
            range_tag = "{}m".format(max_range)
            metric_tag = "_{}".format(metric_type)

            pd.DataFrame.to_csv(metrics, out_path + "/results{}{}{}.csv".format(filter_tag, range_tag, metric_tag))

        return metrics.to_json()


        
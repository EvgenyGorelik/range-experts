_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]


data_root = 'data/nuscenes/'
bev_size_factor = 1  #1
point_cloud_range = [-54.0 * bev_size_factor, -54.0 * bev_size_factor, -5.0, 54.0 * bev_size_factor, 54.0 * bev_size_factor, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
image_scale = 2  # compares to original bevfusion config
image_size = [int(256 * image_scale), int(704 * image_scale)]
image_features = 80 #80
max_epochs = 4

# efficient_conv_bn_eval=["img_backbone"]
# activation_checkpointing = ["img_backbone", "img_neck", "view_transform", "fusion_layer"]

model = dict(
    feature_drop_block_size=3,
    feature_drop_block_rate=0.5,
    data_preprocessor=dict(
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            max_voxels=[120000 * bev_size_factor, 160000 * bev_size_factor])),
    pts_middle_encoder=dict(
        sparse_shape=[1440 * bev_size_factor, 1440 * bev_size_factor, 41]),
    img_backbone=dict(
        _delete_=True,
        type='mmdet.ResNet',
        depth=50,
        out_indices=[1, 2, 3],
        norm_cfg=dict(  # The config of normalization layers.
            type='BN',  # Type of norm layer, usually it is BN or GN
            requires_grad=True),  # Whether to train the gamma and beta in BN
        norm_eval=True,  # Whether to freeze the statistics in BN
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint= 'torchvision://resnet50'  # noqa: E501
        )
    ),
    img_neck=dict(
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)
        ),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=image_features,
        image_size=image_size,
        feature_size=[int(32*image_scale), int(88*image_scale)],
        xbound=[-54.0 * bev_size_factor, 54.0 * bev_size_factor, 0.3],
        ybound=[-54.0 * bev_size_factor, 54.0 * bev_size_factor, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0 * bev_size_factor, 0.5],
        downsample=2),
    fusion_layer=dict(in_channels=[image_features, 256]),
    bbox_head=dict(
        train_cfg=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1440 * bev_size_factor, 1440 * bev_size_factor, 41]),
        test_cfg=dict(
            grid_size=[1440 * bev_size_factor, 1440 * bev_size_factor, 41],
            pc_range=[-54.0 * bev_size_factor, -54.0 * bev_size_factor]),
        bbox_coder=dict(
            pc_range=[-54.0 * bev_size_factor, -54.0 * bev_size_factor],
            post_center_range=[-61.2 * bev_size_factor, -61.2 * bev_size_factor, -10.0, 61.2 * bev_size_factor, 61.2 * bev_size_factor, 10.0]),
        )
    )

backend_args = None

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        resize_lim=[0.38 * image_scale, 0.55 * image_scale],
        final_dim=image_size,
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointSample', num_points=0.01), # 1% to 100% of the points
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # Actually, 'GridMask' is not used here
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=max_epochs,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats', 'lidar2ego'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='ImageAug3D',
        resize_lim=[0.48 * image_scale, 0.48 * image_scale],  # first resize then crop
        final_dim=image_size,
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'lidar2ego'
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        dataset=dict(pipeline=train_pipeline, modality=input_modality)))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, modality=input_modality))
test_dataloader = val_dataloader


optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(type='AdamW', lr=0.0002*0.25, weight_decay=0.01),
        clip_grad=dict(max_norm=35, norm_type=2),
        # accumulative_counts=2
        )

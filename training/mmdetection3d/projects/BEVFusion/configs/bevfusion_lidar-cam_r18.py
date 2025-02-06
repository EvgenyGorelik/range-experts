_base_ = [
    './bevfusion_lidar-cam_r50.py'
]

model = dict(
    img_backbone=dict(
        depth=18,
        init_cfg=dict(
            checkpoint= 'torchvision://resnet18'  # noqa: E501
        )
    ),
    img_neck=dict(
        in_channels=[128, 256, 512],
        )
    )

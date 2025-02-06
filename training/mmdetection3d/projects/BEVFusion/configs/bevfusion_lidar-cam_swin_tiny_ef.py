_base_ = [
    './bevfusion_lidar-cam_swin_tiny.py'
]


model = dict(
    fusion_layer=dict(type='EqualConvFuser', mid_channels=160)
)
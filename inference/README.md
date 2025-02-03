# BEVFusion

BEVFusion ROS node based on [CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion).

Tested with CUDA 11.8 and TensorRT-8.4.2.4

Build BEVFusion model
```
cd path/to/bev_fusion
bash scripts/build_trt_engine.sh
```

You have to link bevfusion to the src directory. Then build the BEVFusion package using colcon.

Required `bag` information:
 - camera images (*optional*)
 - lidar sensor
 - `tf` (lidar->camera, camera->lidar, lidar->map)
 - `camera_info` with camera intrinsics

Launch a bag file with required info:
```
bag play <rosbag> --clock
```

Launch `logging_simulator`:
```
lv logging_simulator.launch.xml launch_detection:=false
```



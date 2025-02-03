#ifndef __BEVFUSION_NODE__HPP__
#define __BEVFUSION_NODE__HPP__

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/opencv.hpp>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/detected_object_kinematics.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include "tier4_autoware_utils/geometry/geometry.hpp"
#include "tier4_autoware_utils/math/constants.hpp"

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#include "NvInfer.h"
#include "bevfusion/bevfusion.hpp"

namespace bevfusion
{
    std::map<size_t, const char *> NUSCENES_CLASS{
        {0, "car"},
        {1, "truck"},
        {2, "construction_vehicle"},
        {3, "bus"},
        {4, "trailer"},
        {5, "barrier"},
        {6, "motorcycle"},
        {7, "bicycle"},
        {8, "pedestrian"},
        {9, "traffic_cone"}};

    std::map<size_t, uint8_t> NUSCENES2AUTOWARE{
        {0, autoware_perception_msgs::msg::ObjectClassification::CAR},        // "car"
        {1, autoware_perception_msgs::msg::ObjectClassification::TRUCK},      // "truck"
        {2, autoware_perception_msgs::msg::ObjectClassification::UNKNOWN},    // "construction_vehicle"
        {3, autoware_perception_msgs::msg::ObjectClassification::BUS},        // "bus"
        {4, autoware_perception_msgs::msg::ObjectClassification::TRAILER},    // "trailer"
        {5, autoware_perception_msgs::msg::ObjectClassification::UNKNOWN},    // "barrier"
        {6, autoware_perception_msgs::msg::ObjectClassification::MOTORCYCLE}, // "motorcycle"
        {7, autoware_perception_msgs::msg::ObjectClassification::BICYCLE},    // "bicycle"
        {8, autoware_perception_msgs::msg::ObjectClassification::PEDESTRIAN}, // "pedestrian"
        {9, autoware_perception_msgs::msg::ObjectClassification::UNKNOWN}};   // "traffic_cone"

    class DataArray
    {
    public:
        float *data;    // data array
        uint32_t size;  // number of points
        uint32_t dim;   // number of dimensions
        uint64_t stamp; // timestamp
        ~DataArray();
    };

    class BEVFusionNode : public rclcpp::Node
    {
    public:
        BEVFusionNode(const rclcpp::NodeOptions &options);

    protected:
        // ######### ROS OBJECTS #########
        std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>> sub_cloud_;
        std::vector<std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>>> camera_subs_compressed_;
        std::vector<std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::Image>>> camera_subs_raw_;
        std::vector<std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>>> camera_info_subs_;

        rclcpp::Publisher<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_;

        tf2_ros::Buffer tf_buffer_;
        tf2_ros::TransformListener tf_listener_;

        std::map<size_t, sensor_msgs::msg::CameraInfo::SharedPtr> camera_info_map_;
        std::map<size_t, sensor_msgs::msg::CompressedImage::SharedPtr> camera_map_compressed_;
        std::map<size_t, sensor_msgs::msg::Image::SharedPtr> camera_map_raw_;
        std::list<std::shared_ptr<DataArray>> lidar_buffer_;

        // ######### NV OBJECTS #########
        std::shared_ptr<Core> core_;
        cudaStream_t stream_;

        // ######### PARAMETERS #########
        bool tf_initialized = false;
        bool core_initialized = false;
        int image_width_;
        int image_height_;
        size_t num_cameras_;
        size_t lidar_buffer_size_;
        size_t max_lidar_points_;
        float intensity_scaling;
        std::string config_;
        std::string lidar_topic_;
        std::vector<std::string> input_camera_info_topics_;
        std::vector<std::string> input_camera_topics_;
        std::string tf_lidar_topic_;
        std::vector<std::string> tf_image_topics_;
        std::string world_frame_;

        // ######### FUNCTIONS #########
        std::shared_ptr<DataArray> extract_pc(sensor_msgs::msg::PointCloud2::SharedPtr);
        std::shared_ptr<DataArray> densify_pc(sensor_msgs::msg::PointCloud2::SharedPtr);
        
        bool init();
        void init_trt_core();
        void cameraInfoCallback(sensor_msgs::msg::CameraInfo::SharedPtr, const std::size_t);
        void cameraCallback_compressed(sensor_msgs::msg::CompressedImage::SharedPtr, const std::size_t);
        void cameraCallback_raw(sensor_msgs::msg::Image::SharedPtr, const std::size_t);
        cv::Mat extract_img_compressed(sensor_msgs::msg::CompressedImage::SharedPtr);
        cv::Mat extract_img_raw(sensor_msgs::msg::Image::SharedPtr);
        void lidarCallback(sensor_msgs::msg::PointCloud2::SharedPtr);
        std::vector<autoware_perception_msgs::msg::DetectedObject> process(std::shared_ptr<DataArray>);

        // ######### DEBUGGING #########
        bool debug_mode;
        std::string export_directory;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub_ptr_;
    };
};

#endif //__BEVFUSION_NODE__HPP__

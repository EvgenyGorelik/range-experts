#include "bevfusion_node.hpp"

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/random_sample.h>
#include <filesystem>

namespace bevfusion
{
    DataArray::~DataArray()
    {
        // delete data array in destructur to avoid memory leak!
        delete data;
    }

    BEVFusionNode::BEVFusionNode(const rclcpp::NodeOptions &options) : rclcpp::Node("data_extraction", options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        RCLCPP_INFO_ONCE(this->get_logger(), "Initializing BEVFusionNode");
        uint32_t slag = declare_parameter<int>("slag", 10);

        num_cameras_ = declare_parameter<int>("num_cameras", 1);
        for (uint32_t camera = 0; camera < num_cameras_; camera++)
        {
            tf_image_topics_.emplace_back(declare_parameter("tf_camera_" + std::to_string(camera), ""));
            input_camera_topics_.emplace_back(declare_parameter("camera_" + std::to_string(camera) + "_topic", ""));
            input_camera_info_topics_.emplace_back(declare_parameter("camera_" + std::to_string(camera) + "_info_topic", ""));
        }

        lidar_topic_ = declare_parameter("lidar_topic", "");
        config_ = declare_parameter("config", "");
        std::function<void(const sensor_msgs::msg::PointCloud2::SharedPtr msg)> lidar_callback_fnc =
            std::bind(&BEVFusionNode::lidarCallback, this, std::placeholders::_1);
        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic_.c_str(), rclcpp::QoS{1}.best_effort(), lidar_callback_fnc);

        // sub camera info
        camera_info_subs_.resize(num_cameras_);
        for (std::size_t cam_i = 0; cam_i < num_cameras_; ++cam_i)
        {
            std::function<void(const sensor_msgs::msg::CameraInfo::SharedPtr msg)> fnc =
                std::bind(&BEVFusionNode::cameraInfoCallback, this, std::placeholders::_1, cam_i);
            camera_info_subs_.at(cam_i) = this->create_subscription<sensor_msgs::msg::CameraInfo>(
                input_camera_info_topics_.at(cam_i), rclcpp::QoS{1}.best_effort(), fnc);
        }

        bool compressed = declare_parameter("compressed", true);
        // sub camera images
        if (compressed) 
            camera_subs_compressed_.resize(num_cameras_);
        else
            camera_subs_raw_.resize(num_cameras_);

        for (std::size_t cam_i = 0; cam_i < num_cameras_; ++cam_i)
        {
            if (compressed) 
            {
                std::function<void(const sensor_msgs::msg::CompressedImage::SharedPtr msg)> fnc =
                    std::bind(&BEVFusionNode::cameraCallback_compressed, this, std::placeholders::_1, cam_i);
                camera_subs_compressed_.at(cam_i) = this->create_subscription<sensor_msgs::msg::CompressedImage>(
                    input_camera_topics_.at(cam_i), rclcpp::QoS{1}.best_effort(), fnc);
            }
            else
            {
                std::function<void(const sensor_msgs::msg::Image::SharedPtr msg)> fnc =
                    std::bind(&BEVFusionNode::cameraCallback_raw, this, std::placeholders::_1, cam_i);
                camera_subs_raw_.at(cam_i) = this->create_subscription<sensor_msgs::msg::Image>(
                    input_camera_topics_.at(cam_i), rclcpp::QoS{1}.best_effort(), fnc);
            }
        }

        std::string objects_pub_topic = declare_parameter("objects_pub_topic", "");
        objects_pub_ = this->create_publisher<autoware_perception_msgs::msg::DetectedObjects>(
            objects_pub_topic.c_str(), rclcpp::QoS{1});

        world_frame_ = declare_parameter("world_frame", "map");

        lidar_buffer_size_ = declare_parameter<int>("lidar_buffer_size", 3);
        max_lidar_points_ = declare_parameter<int>("max_lidar_points", 100000);
        intensity_scaling = declare_parameter<float>("intensity_scaling", 1.0);

        cudaStreamCreate(&stream_);
        RCLCPP_INFO_STREAM(this->get_logger(), "CUDA Stream Initialized");

        debug_mode = declare_parameter("debug", false);
        if (debug_mode)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "################ RUNNING IN DEBUG MODE ################");
            debug_pub_ptr_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug_output", rclcpp::QoS{1});
            export_directory = declare_parameter("export_directory", "");
            RCLCPP_DEBUG_STREAM(this->get_logger(), "num_cameras " << num_cameras_);
            RCLCPP_DEBUG_STREAM(this->get_logger(), "camera names: ");
            for (std::string &camera_name : input_camera_topics_)
                RCLCPP_DEBUG_STREAM(this->get_logger(),  camera_name);
        }
        RCLCPP_INFO_STREAM(this->get_logger(), "Initialization complete");
    }

    void BEVFusionNode::init_trt_core()
    {
        core_ = bevfusion::create_core(config_);
        // std::string pck_path = "/home/tp3/src/bev_fusion";
        // std::string model = "resnet50";
        // std::string precision = "fp16";
        // core_ = bevfusion::create_core(pck_path, model, precision, 1600, 900);
        core_->print();
        fflush(stdout);

        core_->set_capacity_points(max_lidar_points_ * lidar_buffer_size_);
        if (core_ == nullptr)
        {
            printf("Core inititialization failed.\n");
            exit(-1);
        }
        RCLCPP_INFO_STREAM(this->get_logger(), "Core Initialized");
        core_initialized = true;
    }

    std::shared_ptr<DataArray> BEVFusionNode::extract_pc(sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::shared_ptr<pcl::PointCloud<pcl::PointXYZI>> cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        pcl::fromROSMsg(*msg, *cloud);
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Received " << cloud->points.size() << " points");

        if (cloud->points.size() > max_lidar_points_) {
            RCLCPP_WARN_STREAM(this->get_logger(), "Number of points received exceeds network capacity.");
            // Randomly sample 10 points from cloud
            pcl::RandomSample<pcl::PointXYZI> sample (true); // Extract removed indices
            sample.setInputCloud (cloud);
            sample.setSample(max_lidar_points_);

            // Indices
            pcl::Indices indices;
            sample.filter (indices);

            // Cloud
            sample.filter(*cloud);

        }
        std::shared_ptr<DataArray> ld = std::make_shared<DataArray>();
        Eigen::Matrix4f m_lidar2world;
        try
        {   
            if (!strcmp(tf_lidar_topic_.c_str(), ""))
                tf_lidar_topic_ = msg->header.frame_id;
            m_lidar2world = tf2::transformToEigen(tf_buffer_.lookupTransform(
                                                      world_frame_, tf_lidar_topic_, msg->header.stamp, rclcpp::Duration::from_seconds(0.2)))
                                .matrix()
                                .cast<float>();
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN_STREAM(this->get_logger(), ex.what());
            return ld;
        }
        ld->dim = 5;
        ld->size = cloud->points.size();
        ld->stamp = static_cast<uint64_t>(msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec);
        ld->data = new float[cloud->points.size() * ld->dim];

        Eigen::Vector4f point;
        for (size_t i = 0; i < cloud->points.size(); ++i)
        {
            long index = i * 5;
            point = m_lidar2world * Eigen::Vector4f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1);
            ld->data[index] = point(0);
            ld->data[index + 1] = point(1);
            ld->data[index + 2] = point(2);
            ld->data[index + 3] = cloud->points[i].intensity * intensity_scaling;
            ld->data[index + 4] = 0;
        }
        return ld;
    }

    std::shared_ptr<DataArray> BEVFusionNode::densify_pc(sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::shared_ptr<DataArray> ld = std::make_shared<DataArray>();
        Eigen::Matrix4f m_world2lidar;
        try
        {
            m_world2lidar = tf2::transformToEigen(tf_buffer_.lookupTransform(
                                                      tf_lidar_topic_, world_frame_, msg->header.stamp, rclcpp::Duration::from_seconds(0.2)))
                                .matrix()
                                .cast<float>();
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN_STREAM(this->get_logger(), ex.what());
            return ld;
        }
        size_t num_points = 0;
        for (auto &item : lidar_buffer_)
            num_points += item->size;

        ld->dim = 5;
        ld->size = num_points;
        ld->data = new float[num_points * ld->dim];
        ld->stamp = lidar_buffer_.back()->stamp;
        size_t ld_data_pointer = 0;

        pcl::PointCloud<pcl::PointXYZ> debug_out_cloud;
        if (debug_mode)
            debug_out_cloud.points.resize(ld->size);

        for (auto &item : lidar_buffer_)
        {
            for (long i = 0; i < item->size; i++)
            {
                long total_index = ld_data_pointer * 5;
                long index = i * 5;
                Eigen::Vector4f point = m_world2lidar * Eigen::Vector4f(item->data[index], item->data[index + 1], item->data[index + 2], 1);
                ld->data[total_index] = point(0);
                ld->data[total_index + 1] = point(1);
                ld->data[total_index + 2] = point(2);
                ld->data[total_index + 3] = item->data[index + 3];
                ld->data[total_index + 4] = (ld->stamp - item->stamp) / 1e9;
                if (debug_mode)
                {
                    pcl::PointXYZ &debug_point = debug_out_cloud.points[ld_data_pointer];
                    debug_point.x = point(0);
                    debug_point.y = point(1);
                    debug_point.z = point(2);
                }
                ld_data_pointer++;
            }
        }

        if (debug_mode)
        {
            sensor_msgs::msg::PointCloud2 debug_cloud_msg;
            pcl::toROSMsg(debug_out_cloud, debug_cloud_msg);
            debug_cloud_msg.header = msg->header;
            debug_cloud_msg.header.frame_id = world_frame_.c_str();
            debug_pub_ptr_->publish(debug_cloud_msg);
        }
        return ld;
    }

    cv::Mat BEVFusionNode::extract_img_compressed(sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {
        cv::Mat raw_image;
        const std::string &format = msg->format;
        const std::string encoding = format.substr(0, format.find(";"));

        constexpr int DECODE_GRAY = 0;
        constexpr int DECODE_RGB = 1;

        bool encoding_is_bayer = encoding.find("bayer") != std::string::npos;
        if (!encoding_is_bayer)
        {
            raw_image = cv::imdecode(cv::Mat(msg->data), DECODE_RGB);
        }
        else
        {
            raw_image = cv::imdecode(cv::Mat(msg->data), DECODE_GRAY);

            if (encoding == "bayer_rggb8")
            {
                cv::cvtColor(raw_image, raw_image, cv::COLOR_BayerBG2BGR);
            }
            else if (encoding == "bayer_bggr8")
            {
                cv::cvtColor(raw_image, raw_image, cv::COLOR_BayerRG2BGR);
            }
            else if (encoding == "bayer_grbg8")
            {
                cv::cvtColor(raw_image, raw_image, cv::COLOR_BayerGB2BGR);
            }
            else if (encoding == "bayer_gbrg8")
            {
                cv::cvtColor(raw_image, raw_image, cv::COLOR_BayerGR2BGR);
            }
            else
            {
                std::cerr << encoding << " is not supported encoding" << std::endl;
                std::cerr << "Please implement additional decoding in " << __FUNCTION__ << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if (debug_mode && strcmp(export_directory.c_str(), ""))
        {
            std::string filename = export_directory + "/image_" + msg->header.frame_id + "_" + std::to_string(msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec) + ".jpg";
            RCLCPP_DEBUG_STREAM(this->get_logger(), "Saving image as " << filename);
            cv::imwrite(filename, raw_image);
        }
        return raw_image;
    }

    cv::Mat BEVFusionNode::extract_img_raw(sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat raw_image;
        cv_bridge::CvImagePtr in_image_ptr;
        try {
            in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception & e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return raw_image;
        }
        raw_image = in_image_ptr->image;
        return raw_image;
    }

    void BEVFusionNode::cameraInfoCallback(
        const sensor_msgs::msg::CameraInfo::SharedPtr input_camera_info_msg,
        const std::size_t camera_id)
    {
        camera_info_map_[camera_id] = input_camera_info_msg;
    }

    void BEVFusionNode::cameraCallback_compressed(
        const sensor_msgs::msg::CompressedImage::SharedPtr input_camera_msg,
        const std::size_t camera_id)
    {
        camera_map_compressed_[camera_id] = input_camera_msg;
    }

    void BEVFusionNode::cameraCallback_raw(
        const sensor_msgs::msg::Image::SharedPtr input_camera_msg,
        const std::size_t camera_id)
    {
        camera_map_raw_[camera_id] = input_camera_msg;
    }

    void BEVFusionNode::lidarCallback(sensor_msgs::msg::PointCloud2::SharedPtr pc_msg)
    {
        std::shared_ptr<DataArray> ld = extract_pc(pc_msg);
        if (ld->dim == 0)
            return;
        // fill pointcloud buffer
        lidar_buffer_.emplace_back(ld);
        if (lidar_buffer_.size() >= lidar_buffer_size_)
        {
            std::shared_ptr<DataArray> pc_data = densify_pc(pc_msg);
            std::vector<autoware_perception_msgs::msg::DetectedObject> detected_objects = process(pc_data);

            if (detected_objects.size() > 0)
            {
                autoware_perception_msgs::msg::DetectedObjects output_msg;
                output_msg.header = pc_msg->header;
                output_msg.objects = detected_objects;
                objects_pub_->publish(output_msg);
            }
            // pop oldest lidar data
            lidar_buffer_.pop_front();
        }
    }

    std::vector<autoware_perception_msgs::msg::DetectedObject> BEVFusionNode::process(std::shared_ptr<DataArray> ld)
    {
        std::vector<autoware_perception_msgs::msg::DetectedObject> detected_objects;
        if (!init() || ld->dim == 0)
            return detected_objects;

        nv::Tensor lidar_data = nv::Tensor::from_data_reference((void *)ld->data, std::vector<int64_t>{ld->size, ld->dim}, nv::DataType::Float32, false);
        lidar_data = lidar_data.to_device().to_half().to_host();

        int width = image_width_;
        int height = image_height_;
        int channels = 3;

        std::vector<unsigned char *> _images;

        for (std::size_t cam_i = 0; cam_i < 6; ++cam_i)
        {
            cv::Mat img;
            std::vector<unsigned char> buffer;
            if (camera_map_compressed_.find(cam_i) == camera_map_compressed_.end())
            {
                if (camera_map_raw_.find(cam_i) == camera_map_raw_.end()) {
                    RCLCPP_WARN_STREAM(this->get_logger(), "no camera image. id is " << cam_i);
                    // fill zero matrix if necessary...
                    img = cv::Mat::zeros(cv::Size(image_width_, image_height_), CV_8UC3);
                }
                else
                {
                    img = extract_img_raw(camera_map_raw_.at(cam_i));
                }    
            }
            else
            {
                img = extract_img_compressed(camera_map_compressed_.at(cam_i));
            }
            _images.emplace_back(img.data);
            // cv::imencode(".jpg", img, buffer);
            // unsigned char * img_arr = stbi_load_from_memory(buffer.data(), buffer.size(), &width, &height, &channels, 0); // HUGE MEMORY LEAK IN THIS FUNCTION...
            // _images.emplace_back(img_arr);
        }
        std::vector<head::transbbox::BoundingBox> bboxes = core_->forward((const unsigned char **)_images.data(), lidar_data.ptr<nvtype::half>(), lidar_data.size(0), stream_);

        // release lidar data after usage
        lidar_data.release();

        RCLCPP_DEBUG_STREAM(this->get_logger(), "Calculated " << bboxes.size() << " Bounding Boxes");

        bool has_twist = true;

        detected_objects.reserve(bboxes.size());

        std::ofstream debug_bb_file;
        if (debug_mode && strcmp(export_directory.c_str(), ""))
        {
            std::string bb_filename = export_directory + "/bboxes_" + std::to_string(ld->stamp) + ".csv";
            std::string pc_filename = export_directory + "/pc_" + std::to_string(ld->stamp) + ".tensor";
            lidar_data.save(pc_filename);
            debug_bb_file = std::ofstream(bb_filename);
        }

        for (head::transbbox::BoundingBox bbox : bboxes)
        {
            if (debug_mode)
            {
                std::string bb_info = std::to_string(bbox.id) + "," +
                                      std::to_string(bbox.position.x) + "," +
                                      std::to_string(bbox.position.y) + "," +
                                      std::to_string(bbox.position.z) + "," +
                                      std::to_string(bbox.size.w) + "," +
                                      std::to_string(bbox.size.l) + "," +
                                      std::to_string(bbox.size.h) + "," +
                                      std::to_string(bbox.z_rotation) + "," +
                                      std::to_string(bbox.score);
                RCLCPP_DEBUG_STREAM(this->get_logger() , NUSCENES_CLASS[bbox.id] << ": (" << bb_info << ")");
                debug_bb_file << bb_info << "\n";
            }

            autoware_perception_msgs::msg::DetectedObject obj;

            // classification
            autoware_perception_msgs::msg::ObjectClassification classification;
            classification.probability = 1.0f;
            if (NUSCENES2AUTOWARE.find(bbox.id) != NUSCENES2AUTOWARE.end())
            {
                classification.label = NUSCENES2AUTOWARE[bbox.id];
            }
            else
            {
                classification.label = autoware_perception_msgs::msg::ObjectClassification::UNKNOWN;
                RCLCPP_WARN_STREAM(this->get_logger(), "Unexpected label: UNKNOWN is set.");
            }

            obj.existence_probability = bbox.score;
            obj.classification.emplace_back(classification);

            // pose and shape
            // mmdet3d yaw format to ros yaw format
            obj.kinematics.pose_with_covariance.pose.position =
                tier4_autoware_utils::createPoint(bbox.position.x, bbox.position.y, bbox.position.z);

            float yaw = bbox.z_rotation + tier4_autoware_utils::pi / 2;
            obj.kinematics.pose_with_covariance.pose.orientation.z = -std::sin(yaw / 2);
            obj.kinematics.pose_with_covariance.pose.orientation.w = std::cos(yaw / 2);

            obj.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
            obj.shape.dimensions =
                tier4_autoware_utils::createTranslation(bbox.size.l, bbox.size.w, bbox.size.h);

            // twist
            if (has_twist)
            {
                float vel_x = bbox.velocity.vx;
                float vel_y = bbox.velocity.vy;
                geometry_msgs::msg::Twist twist;
                twist.linear.x = std::sqrt(std::pow(vel_x, 2) + std::pow(vel_y, 2));
                twist.angular.z = 2 * (std::atan2(vel_y, vel_x) - yaw);
                obj.kinematics.twist_with_covariance.twist = twist;
                obj.kinematics.has_twist = has_twist;
                obj.kinematics.has_twist_covariance = false;
            }
            detected_objects.emplace_back(obj);
        }
        if (debug_mode)
        {
            // Close the file
            debug_bb_file.close();
        }
        return detected_objects;
    }

    bool BEVFusionNode::init()
    {
        if (tf_initialized)
            return true;
        if (!strcmp(tf_lidar_topic_.c_str(), ""))
            return false;
        RCLCPP_INFO_STREAM(this->get_logger(), "Initializing TFs");
        std::vector<int32_t> shape({1, 6, 4, 4});

        Eigen::MatrixXf ident = Eigen::MatrixXf::Identity(shape[2], shape[3]);

        float *cam2lidar = new float[6 * shape[2] * shape[3]];
        float *lidar2cam = new float[6 * shape[2] * shape[3]];
        float *img_aug_matrix = new float[6 * shape[2] * shape[3]];
        float *cam_intrinsics = new float[6 * shape[2] * shape[3]];

        for (uint32_t cam_idx = 0; cam_idx < num_cameras_; ++cam_idx)
        {
            try
            {
                Eigen::MatrixXf m_cam_intrinsics = Eigen::MatrixXf::Identity(4, 4);
                if (camera_info_map_.find(cam_idx) == camera_info_map_.end())
                {
                    RCLCPP_WARN_STREAM(this->get_logger(), "no camera info for camera id " << cam_idx);
                    return false;
                }
                else
                {
                    const Eigen::Matrix3d Kd_t = Eigen::Map<const Eigen::Matrix<double, 3, 3>>(camera_info_map_.at(cam_idx)->k.data());
                    m_cam_intrinsics(Eigen::seqN(0, 3), Eigen::seqN(0, 3)) = Kd_t.cast<float>().transpose();
                    image_height_ = camera_info_map_.at(cam_idx)->height;
                    image_width_ = camera_info_map_.at(cam_idx)->width;
                }

                Eigen::Matrix4f m_cam2lidar = tf2::transformToEigen(tf_buffer_.lookupTransform(
                                                                        tf_lidar_topic_, tf_image_topics_.at(cam_idx), rclcpp::Clock{}.now(), rclcpp::Duration::from_seconds(0.01)))
                                                  .matrix()
                                                  .cast<float>();
                Eigen::Matrix4f m_lidar2cam = tf2::transformToEigen(tf_buffer_.lookupTransform(
                                                                        tf_image_topics_.at(cam_idx), tf_lidar_topic_, rclcpp::Clock{}.now(), rclcpp::Duration::from_seconds(0.01)))
                                                  .matrix()
                                                  .cast<float>();
                // fill data in COLUMN_MAJOR
                for (int i = 0; i < shape[2] * shape[3]; i++)
                {
                    cam2lidar[cam_idx * shape[2] * shape[3] + i] = m_cam2lidar(i / m_lidar2cam.cols(), i % m_lidar2cam.cols());
                    lidar2cam[cam_idx * shape[2] * shape[3] + i] = m_lidar2cam(i / m_lidar2cam.cols(), i % m_lidar2cam.cols());
                    cam_intrinsics[cam_idx * shape[2] * shape[3] + i] = m_cam_intrinsics(i / m_lidar2cam.cols(), i % m_lidar2cam.cols());
                    img_aug_matrix[cam_idx * shape[2] * shape[3] + i] = ident(i / m_lidar2cam.cols(), i % m_lidar2cam.cols());
                }
            }
            catch (tf2::TransformException &ex)
            {
                RCLCPP_WARN_STREAM(this->get_logger(), ex.what());
            }
        }

        if (!core_initialized)
            init_trt_core();

        nv::Tensor cam2lidar_nv = nv::Tensor::from_data(cam2lidar, shape, nv::DataType::Float32, false, stream_);
        nv::Tensor lidar2cam_nv = nv::Tensor::from_data(lidar2cam, shape, nv::DataType::Float32, false, stream_);
        nv::Tensor img_aug_matrix_nv = nv::Tensor::from_data(img_aug_matrix, shape, nv::DataType::Float32, false, stream_);
        nv::Tensor cam_intrinsics_nv = nv::Tensor::from_data(cam_intrinsics, shape, nv::DataType::Float32, false, stream_);

        core_->update(cam2lidar_nv.ptr<float>(), cam_intrinsics_nv.ptr<float>(), lidar2cam_nv.ptr<float>(), img_aug_matrix_nv.ptr<float>(), stream_);

        RCLCPP_INFO_STREAM(this->get_logger(), "Initialized TFs");
        tf_initialized = true;
        return true;
    }
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(bevfusion::BEVFusionNode)
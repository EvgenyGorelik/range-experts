#include "yaml-cpp/yaml.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "common/dtype.hpp"
#include "common/tensor.hpp"


int main()
{
    YAML::Node config = YAML::LoadFile("config/bevfusion.param.yaml");

    std::string pck_path = config["model_path"].as<std::string>();
    std::string model = config["model_name"].as<std::string>();

    int normalization_image_width = config["normalization"]["image_width"].as<int>();
    int normalization_image_height = config["normalization"]["image_height"].as<int>();
    int output_width = config["normalization"]["output_width"].as<int>();
    int output_height = config["normalization"]["output_height"].as<int>();
    int normalization_num_camera = config["normalization"]["num_camera"].as<int>();
    float resize_lim = config["normalization"]["resize_lim"].as<float>();
    size_t interpolation = static_cast<size_t>(config["normalization"]["interpolation"].as<size_t>());

    float mean[3];
    float std[3];

    const YAML::Node& mean_node = config["normalization"]["mean"];
    const YAML::Node& std_node = config["normalization"]["std"];
    for(unsigned i = 0; i < std_node.size(); i++) {
        mean[i] = mean_node[i].as<float>();
        std[i] = std_node[i].as<float>();
    }

    const YAML::Node& voxelization_min_range = config["voxelization"]["min_range"];
    const YAML::Node& voxelization_max_range = config["voxelization"]["max_range"];
    const YAML::Node& voxelization_voxel_size = config["voxelization"]["voxel_size"];
    
    nvtype::Float3 min_range = nvtype::Float3(voxelization_min_range[0].as<float>(), voxelization_min_range[1].as<float>(), voxelization_min_range[2].as<float>());
    nvtype::Float3 max_range = nvtype::Float3(voxelization_max_range[0].as<float>(), voxelization_max_range[1].as<float>(), voxelization_max_range[2].as<float>());
    nvtype::Float3 voxelization_voxel_size_nv = nvtype::Float3(voxelization_voxel_size[0].as<float>(), voxelization_voxel_size[1].as<float>(), voxelization_voxel_size[2].as<float>());
    int max_points_per_voxel = config["voxelization"]["max_points_per_voxel"].as<int>();
    int max_points = config["voxelization"]["max_points"].as<int>();
    int max_voxels = config["voxelization"]["max_voxels"].as<int>();
    int num_feature = config["voxelization"]["num_feature"].as<int>();

    std::string lidar_model = nv::format("%s/model/%s/lidar.backbone.xyz.onnx", config["model_path"].as<std::string>().c_str(),  config["model_name"].as<std::string>().c_str());
    size_t precision = static_cast<size_t>(config["precision"].as<int>());

    const YAML::Node& geometry_xbound_node = config["geometry"]["xbound"];
    const YAML::Node& geometry_ybound_node = config["geometry"]["ybound"];
    const YAML::Node& geometry_zbound_node = config["geometry"]["zbound"];
    const YAML::Node& geometry_dbound_node = config["geometry"]["dbound"];
    const YAML::Node& geometry_dim_node = config["geometry"]["geometry_dim"];

    nvtype::Float3 xbound = nvtype::Float3(geometry_xbound_node[0].as<float>(), geometry_xbound_node[1].as<float>(), geometry_xbound_node[2].as<float>());
    nvtype::Float3 ybound = nvtype::Float3(geometry_ybound_node[0].as<float>(), geometry_ybound_node[1].as<float>(), geometry_ybound_node[2].as<float>());
    nvtype::Float3 zbound = nvtype::Float3(geometry_zbound_node[0].as<float>(), geometry_zbound_node[1].as<float>(), geometry_zbound_node[2].as<float>());
    nvtype::Float3 dbound = nvtype::Float3(geometry_dbound_node[0].as<float>(), geometry_dbound_node[1].as<float>(), geometry_dbound_node[2].as<float>());
    int geometry_image_width = config["geometry"]["image_width"].as<int>();
    int geometry_image_height = config["geometry"]["image_height"].as<int>();
    int geometry_feat_width = config["geometry"]["feat_width"].as<int>();
    int geometry_feat_height = config["geometry"]["feat_height"].as<int>();
    int geometry_num_camera = config["geometry"]["num_camera"].as<int>();
    nvtype::Int3 geometry_dim = nvtype::Int3(geometry_dim_node[0].as<int>(), geometry_dim_node[1].as<int>(), geometry_dim_node[2].as<int>());

    const YAML::Node& transbbox_pc_range = config["transbbox"]["pc_range"];
    const YAML::Node& transbbox_post_center_range_start = config["transbbox"]["post_center_range_start"];
    const YAML::Node& transbbox_post_center_range_end = config["transbbox"]["post_center_range_end"];
    const YAML::Node& transbbox_voxel_size = config["transbbox"]["voxel_size"];

    int out_size_factor = config["transbbox"]["out_size_factor"].as<int>();
    nvtype::Float2 pc_range = nvtype::Float2(transbbox_pc_range[0].as<float>(), transbbox_pc_range[1].as<float>());
    nvtype::Float3 post_center_range_start = nvtype::Float3(transbbox_post_center_range_start[0].as<float>(), transbbox_post_center_range_start[1].as<float>(), transbbox_post_center_range_start[2].as<float>());
    nvtype::Float3 post_center_range_end = nvtype::Float3(transbbox_post_center_range_end[0].as<float>(), transbbox_post_center_range_end[1].as<float>(), transbbox_post_center_range_end[2].as<float>());
    nvtype::Float2 voxel_size = nvtype::Float2(transbbox_voxel_size[0].as<float>(), transbbox_voxel_size[1].as<float>());
    std::string head_model = nv::format("%s/model/%s/build/head.bbox.plan",  config["model_path"].as<std::string>().c_str(),  config["model_name"].as<std::string>().c_str());
    float confidence_threshold = config["transbbox"]["confidence_threshold"].as<float>();
    bool sorted_bboxes = config["transbbox"]["sorted_bboxes"].as<bool>();

    std::string camera_model = nv::format("%s/model/%s/build/camera.backbone.plan", config["model_path"].as<std::string>().c_str(),  config["model_name"].as<std::string>().c_str());
    std::string transfusion = nv::format("%s/model/%s/build/fuser.plan",  config["model_path"].as<std::string>().c_str(),  config["model_name"].as<std::string>().c_str());
    std::string camera_vtransform = nv::format("%s/model/%s/build/camera.vtransform.plan",  config["model_path"].as<std::string>().c_str(),  config["model_name"].as<std::string>().c_str());

    return 0;
}
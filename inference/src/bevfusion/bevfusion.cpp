/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated configumentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "bevfusion.hpp"

#include <numeric>

#include "common/check.hpp"
#include "common/timer.hpp"

#include "yaml-cpp/yaml.h"
#include <fstream>

namespace bevfusion {

class CoreImplement : public Core {
 public:
  virtual ~CoreImplement() {
    if (lidar_points_device_) checkRuntime(cudaFree(lidar_points_device_));
    if (lidar_points_host_) checkRuntime(cudaFreeHost(lidar_points_host_));
  }

  bool init(const CoreParameter& param) {
    camera_backbone_ = camera::create_backbone(param.camera_model);
    if (camera_backbone_ == nullptr) {
      printf("Failed to create camera backbone.\n");
      return false;
    }

    camera_bevpool_ =
        camera::create_bevpool(camera_backbone_->camera_shape(), param.geometry.geometry_dim.x, param.geometry.geometry_dim.y);
    if (camera_bevpool_ == nullptr) {
      printf("Failed to create camera bevpool.\n");
      return false;
    }

    camera_vtransform_ = camera::create_vtransform(param.camera_vtransform);
    if (camera_vtransform_ == nullptr) {
      printf("Failed to create camera vtransform.\n");
      return false;
    }

    transfusion_ = fuser::create_transfusion(param.transfusion);
    if (transfusion_ == nullptr) {
      printf("Failed to create transfusion.\n");
      return false;
    }

    transbbox_ = head::transbbox::create_transbbox(param.transbbox);
    if (transbbox_ == nullptr) {
      printf("Failed to create head transbbox.\n");
      return false;
    }

    lidar_scn_ = lidar::create_scn(param.lidar_scn);
    if (lidar_scn_ == nullptr) {
      printf("Failed to create lidar scn.\n");
      return false;
    }

    normalizer_ = camera::create_normalization(param.normalize);
    if (normalizer_ == nullptr) {
      printf("Failed to create normalizer.\n");
      return false;
    }

    camera_depth_ = camera::create_depth(param.normalize.output_width, param.normalize.output_height, param.normalize.num_camera);
    if (camera_depth_ == nullptr) {
      printf("Failed to create depth.\n");
      return false;
    }

    camera_geometry_ = camera::create_geometry(param.geometry);
    if (camera_geometry_ == nullptr) {
      printf("Failed to create geometry.\n");
      return false;
    }

    param_ = param;
    set_capacity_points(300000);
    return true;
  }

  std::vector<head::transbbox::BoundingBox> forward_only(const void* camera_images, const nvtype::half* lidar_points,
                                                         int num_points, void* stream, bool do_normalization) {
    int cappoints = static_cast<int>(capacity_points_);
    if (num_points > cappoints) {
      printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
    }

    num_points = std::min(cappoints, num_points);

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    size_t bytes_points = num_points * param_.lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));

    const nvtype::half* lidar_feature = this->lidar_scn_->forward(lidar_points_device_, num_points, stream);
    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);
    }
    const nvtype::half* depth = this->camera_depth_->forward(lidar_points_device_, num_points, 5, stream);

    this->camera_backbone_->forward(normed_images, depth, stream);
    const nvtype::half* camera_bev = this->camera_bevpool_->forward(
        this->camera_backbone_->feature(), this->camera_backbone_->depth(), this->camera_geometry_->indices(),
        this->camera_geometry_->intervals(), this->camera_geometry_->num_intervals(), stream);

    const nvtype::half* camera_bevfeat = camera_vtransform_->forward(camera_bev, stream);
    const nvtype::half* fusion_feature = this->transfusion_->forward(camera_bevfeat, lidar_feature, stream);
    return this->transbbox_->forward(fusion_feature, param_.transbbox.confidence_threshold, stream,
                                     param_.transbbox.sorted_bboxes);
  }

  std::vector<head::transbbox::BoundingBox> forward_timer(const void* camera_images, const nvtype::half* lidar_points,
                                                          int num_points, void* stream, bool do_normalization) {
    int cappoints = static_cast<int>(capacity_points_);
    if (num_points > cappoints) {
      printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
    }

    num_points = std::min(cappoints, num_points);

    printf("==================BEVFusion===================\n");
    std::vector<float> times;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    timer_.start(_stream);

    size_t bytes_points = num_points * param_.lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));
    timer_.stop("[NoSt] CopyLidar");

    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      timer_.start(_stream);
      normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);
      timer_.stop("[NoSt] ImageNrom");
    }

    timer_.start(_stream);
    const nvtype::half* lidar_feature = this->lidar_scn_->forward(lidar_points_device_, num_points, stream);
    times.emplace_back(timer_.stop("Lidar Backbone"));

    timer_.start(_stream);
    const nvtype::half* depth = this->camera_depth_->forward(lidar_points_device_, num_points, 5, stream);
    times.emplace_back(timer_.stop("Camera Depth"));

    timer_.start(_stream);
    this->camera_backbone_->forward(normed_images, depth, stream);
    times.emplace_back(timer_.stop("Camera Backbone"));

    timer_.start(_stream);
    const nvtype::half* camera_bev = this->camera_bevpool_->forward(
        this->camera_backbone_->feature(), this->camera_backbone_->depth(), this->camera_geometry_->indices(),
        this->camera_geometry_->intervals(), this->camera_geometry_->num_intervals(), stream);
    times.emplace_back(timer_.stop("Camera Bevpool"));

    timer_.start(_stream);
    const nvtype::half* camera_bevfeat = camera_vtransform_->forward(camera_bev, stream);
    times.emplace_back(timer_.stop("VTransform"));

    timer_.start(_stream);
    const nvtype::half* fusion_feature = this->transfusion_->forward(camera_bevfeat, lidar_feature, stream);
    times.emplace_back(timer_.stop("Transfusion"));

    timer_.start(_stream);
    auto output =
        this->transbbox_->forward(fusion_feature, param_.transbbox.confidence_threshold, stream, param_.transbbox.sorted_bboxes);
    times.emplace_back(timer_.stop("Head BoundingBox"));

    float total_time = std::accumulate(times.begin(), times.end(), 0.0f, std::plus<float>{});
    printf("Total: %.3f ms\n", total_time);
    printf("=============================================\n");
    return output;
  }

  virtual std::vector<head::transbbox::BoundingBox> forward(const unsigned char** camera_images, const nvtype::half* lidar_points,
                                                            int num_points, void* stream) override {
    if (enable_timer_) {
      return this->forward_timer(camera_images, lidar_points, num_points, stream, true);
    } else {
      return this->forward_only(camera_images, lidar_points, num_points, stream, true);
    }
  }

  virtual std::vector<head::transbbox::BoundingBox> forward_no_normalize(const nvtype::half* camera_normed_images_device,
                                                                         const nvtype::half* lidar_points, int num_points,
                                                                         void* stream) override {
    if (enable_timer_) {
      return this->forward_timer(camera_normed_images_device, lidar_points, num_points, stream, false);
    } else {
      return this->forward_only(camera_normed_images_device, lidar_points, num_points, stream, false);
    }
  }

  virtual void set_timer(bool enable) override { enable_timer_ = enable; }
  virtual void set_capacity_points(size_t capacity) override {
    if (capacity_points_ == capacity) return;

    if (capacity_points_ != 0) {
      if (lidar_points_device_ != nullptr) checkRuntime(cudaFree(lidar_points_device_));
      if (lidar_points_host_ != nullptr) checkRuntime(cudaFreeHost(lidar_points_host_));
    }
    
    capacity_points_ = capacity;
    bytes_capacity_points_ = capacity_points_ * param_.lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMalloc(&lidar_points_device_, bytes_capacity_points_));
    checkRuntime(cudaMallocHost(&lidar_points_host_, bytes_capacity_points_));
  }

  virtual void print() override {
    camera_backbone_->print();
    camera_vtransform_->print();
    transfusion_->print();
    transbbox_->print();
  }

  virtual void update(const float* camera2lidar, const float* camera_intrinsics, const float* lidar2image,
                      const float* img_aug_matrix, void* stream) override {
    camera_depth_->update(img_aug_matrix, lidar2image, stream);
    camera_geometry_->update(camera2lidar, camera_intrinsics, img_aug_matrix, stream);
  }

  virtual void free_excess_memory() override { camera_geometry_->free_excess_memory(); }

 private:
  CoreParameter param_;
  nv::EventTimer timer_;
  nvtype::half* lidar_points_device_ = nullptr;
  nvtype::half* lidar_points_host_ = nullptr;
  size_t capacity_points_ = 0;
  size_t bytes_capacity_points_ = 0;

  std::shared_ptr<camera::Normalization> normalizer_;
  std::shared_ptr<camera::Backbone> camera_backbone_;
  std::shared_ptr<camera::BEVPool> camera_bevpool_;
  std::shared_ptr<camera::VTransform> camera_vtransform_;
  std::shared_ptr<camera::Depth> camera_depth_;
  std::shared_ptr<camera::Geometry> camera_geometry_;
  std::shared_ptr<lidar::SCN> lidar_scn_;
  std::shared_ptr<fuser::Transfusion> transfusion_;
  std::shared_ptr<head::transbbox::TransBBox> transbbox_;
  float confidence_threshold_ = 0;
  bool enable_timer_ = false;
};

std::shared_ptr<Core> create_core(const CoreParameter& param) {
  std::shared_ptr<CoreImplement> instance(new CoreImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}



std::shared_ptr<bevfusion::Core> create_core(const std::string& pck_path, const std::string& model, const std::string& precision, int image_width, int image_height) { //

  printf("Create by %s, %s\n", model.c_str(), precision.c_str());
  bevfusion::camera::NormalizationParameter normalization;
  normalization.image_width = image_width;
  normalization.image_height = image_height;
  normalization.output_width = 704;
  normalization.output_height = 256;
  normalization.num_camera = 6;
  normalization.resize_lim = 0.48f;
  normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;

  float mean[3] = {0.485, 0.456, 0.406};
  float std[3] = {0.229, 0.224, 0.225};
  normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

  bevfusion::lidar::VoxelizationParameter voxelization;
  voxelization.min_range = nvtype::Float3(-54.0f, -54.0f, -5.0);
  voxelization.max_range = nvtype::Float3(+54.0f, +54.0f, +3.0);
  voxelization.voxel_size = nvtype::Float3(0.075f, 0.075f, 0.2f);
  voxelization.grid_size =
      voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
  voxelization.max_points_per_voxel = 10;
  voxelization.max_points = 300000;
  voxelization.max_voxels = 160000;
  voxelization.num_feature = 5;

  bevfusion::lidar::SCNParameter scn;
  scn.voxelization = voxelization;
  scn.model = nv::format("%s/model/%s/lidar.backbone.xyz.onnx", pck_path.c_str(),  model.c_str());
  scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

  if (precision == "int8") {
    scn.precision = bevfusion::lidar::Precision::Int8;
  } else {
    scn.precision = bevfusion::lidar::Precision::Float16;
  }

  bevfusion::camera::GeometryParameter geometry;
  geometry.xbound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.ybound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
  geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);
  geometry.image_width = 704;
  geometry.image_height = 256;
  geometry.feat_width = 88;
  geometry.feat_height = 32;
  geometry.num_camera = 6;
  geometry.geometry_dim = nvtype::Int3(360, 360, 80);

  bevfusion::head::transbbox::TransBBoxParameter transbbox;
  transbbox.out_size_factor = 8;
  transbbox.pc_range = {-54.0f, -54.0f};
  transbbox.post_center_range_start = {-61.2, -61.2, -10.0};
  transbbox.post_center_range_end = {61.2, 61.2, 10.0};
  transbbox.voxel_size = {0.075, 0.075};
  transbbox.model = nv::format("%s/model/%s/build/head.bbox.plan", pck_path.c_str(),  model.c_str());
  transbbox.confidence_threshold = 0.12f;
  transbbox.sorted_bboxes = true;

  bevfusion::CoreParameter param;
  param.camera_model = nv::format("%s/model/%s/build/camera.backbone.plan", pck_path.c_str(),  model.c_str());
  param.normalize = normalization;
  param.lidar_scn = scn;
  param.geometry = geometry;
  param.transfusion = nv::format("%s/model/%s/build/fuser.plan", pck_path.c_str(),  model.c_str());
  param.transbbox = transbbox;
  param.camera_vtransform = nv::format("%s/model/%s/build/camera.vtransform.plan", pck_path.c_str(),  model.c_str());
  return bevfusion::create_core(param);
}

std::shared_ptr<bevfusion::Core> create_core(const std::string& config_file) { 
  
  YAML::Node config = YAML::LoadFile(config_file);

  const std::string& pck_path = config["model_path"].as<std::string>();
  const std::string& model = config["model_name"].as<std::string>();

  bevfusion::camera::NormalizationParameter normalization;
  normalization.image_width = config["normalization"]["image_width"].as<int>();
  normalization.image_height = config["normalization"]["image_height"].as<int>();
  normalization.output_width = config["normalization"]["output_width"].as<int>();
  normalization.output_height = config["normalization"]["output_height"].as<int>();
  normalization.num_camera = config["normalization"]["num_camera"].as<int>();
  normalization.resize_lim = config["normalization"]["resize_lim"].as<float>();
  normalization.interpolation = static_cast<bevfusion::camera::Interpolation>(config["normalization"]["interpolation"].as<size_t>());

  float mean[3];
  float std[3];

  const YAML::Node& mean_node = config["normalization"]["mean"];
  const YAML::Node& std_node = config["normalization"]["std"];
  for(unsigned i = 0; i < std_node.size(); i++) {
    mean[i] = mean_node[i].as<float>();
    std[i] = std_node[i].as<float>();
  }

  normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

  bevfusion::lidar::VoxelizationParameter voxelization;
  const YAML::Node& voxelization_min_range = config["voxelization"]["min_range"];
  const YAML::Node& voxelization_max_range = config["voxelization"]["max_range"];
  const YAML::Node& voxelization_voxel_size = config["voxelization"]["voxel_size"];
  
  voxelization.min_range = nvtype::Float3(voxelization_min_range[0].as<float>(), voxelization_min_range[1].as<float>(), voxelization_min_range[2].as<float>());
  voxelization.max_range = nvtype::Float3(voxelization_max_range[0].as<float>(), voxelization_max_range[1].as<float>(), voxelization_max_range[2].as<float>());
  voxelization.voxel_size = nvtype::Float3(voxelization_voxel_size[0].as<float>(), voxelization_voxel_size[1].as<float>(), voxelization_voxel_size[2].as<float>());
  voxelization.grid_size =
      voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
  voxelization.max_points_per_voxel = config["voxelization"]["max_points_per_voxel"].as<int>();
  voxelization.max_points = config["voxelization"]["max_points"].as<int>();
  voxelization.max_voxels = config["voxelization"]["max_voxels"].as<int>();
  voxelization.num_feature = config["voxelization"]["num_feature"].as<int>();

  bevfusion::lidar::SCNParameter scn;
  scn.voxelization = voxelization;
  scn.model = nv::format("%s/model/%s/lidar.backbone.xyz.onnx", config["model_path"].as<std::string>().c_str(),  config["model_name"].as<std::string>().c_str());
  scn.order = bevfusion::lidar::CoordinateOrder::XYZ;
  scn.precision = static_cast<bevfusion::lidar::Precision>(config["precision"].as<int>());

  bevfusion::camera::GeometryParameter geometry;
  const YAML::Node& xbound = config["geometry"]["xbound"];
  const YAML::Node& ybound = config["geometry"]["ybound"];
  const YAML::Node& zbound = config["geometry"]["zbound"];
  const YAML::Node& dbound = config["geometry"]["dbound"];
  const YAML::Node& geometry_dim = config["geometry"]["geometry_dim"];


  geometry.xbound = nvtype::Float3(xbound[0].as<float>(), xbound[1].as<float>(), xbound[2].as<float>());
  geometry.ybound = nvtype::Float3(ybound[0].as<float>(), ybound[1].as<float>(), ybound[2].as<float>());
  geometry.zbound = nvtype::Float3(zbound[0].as<float>(), zbound[1].as<float>(), zbound[2].as<float>());
  geometry.dbound = nvtype::Float3(dbound[0].as<float>(), dbound[1].as<float>(), dbound[2].as<float>());
  geometry.image_width = config["geometry"]["image_width"].as<int>();
  geometry.image_height = config["geometry"]["image_height"].as<int>();
  geometry.feat_width = config["geometry"]["feat_width"].as<int>();
  geometry.feat_height = config["geometry"]["feat_height"].as<int>();
  geometry.num_camera = config["geometry"]["num_camera"].as<int>();
  geometry.geometry_dim = nvtype::Int3(geometry_dim[0].as<int>(), geometry_dim[1].as<int>(), geometry_dim[2].as<int>());

  bevfusion::head::transbbox::TransBBoxParameter transbbox;

  const YAML::Node& transbbox_pc_range = config["transbbox"]["pc_range"];
  const YAML::Node& transbbox_post_center_range_start = config["transbbox"]["post_center_range_start"];
  const YAML::Node& transbbox_post_center_range_end = config["transbbox"]["post_center_range_end"];
  const YAML::Node& transbbox_voxel_size = config["transbbox"]["voxel_size"];

  transbbox.out_size_factor = config["transbbox"]["out_size_factor"].as<int>();
  transbbox.pc_range = nvtype::Float2(transbbox_pc_range[0].as<float>(), transbbox_pc_range[1].as<float>());
  transbbox.post_center_range_start = nvtype::Float3(transbbox_post_center_range_start[0].as<float>(), transbbox_post_center_range_start[1].as<float>(), transbbox_post_center_range_start[2].as<float>());
  transbbox.post_center_range_end = nvtype::Float3(transbbox_post_center_range_end[0].as<float>(), transbbox_post_center_range_end[1].as<float>(), transbbox_post_center_range_end[2].as<float>());
  transbbox.voxel_size = nvtype::Float2(transbbox_voxel_size[0].as<float>(), transbbox_voxel_size[1].as<float>());
  transbbox.model = nv::format("%s/model/%s/build/head.bbox.plan",  config["model_path"].as<std::string>().c_str(),  config["model_name"].as<std::string>().c_str());
  transbbox.confidence_threshold = config["transbbox"]["confidence_threshold"].as<float>();
  transbbox.sorted_bboxes = config["transbbox"]["sorted_bboxes"].as<bool>();

  bevfusion::CoreParameter param;
  param.camera_model = nv::format("%s/model/%s/build/camera.backbone.plan", config["model_path"].as<std::string>().c_str(),  config["model_name"].as<std::string>().c_str());
  param.normalize = normalization;
  param.lidar_scn = scn;
  param.geometry = geometry;
  param.transfusion = nv::format("%s/model/%s/build/fuser.plan",  config["model_path"].as<std::string>().c_str(),  config["model_name"].as<std::string>().c_str());
  param.transbbox = transbbox;
  param.camera_vtransform = nv::format("%s/model/%s/build/camera.vtransform.plan",  config["model_path"].as<std::string>().c_str(),  config["model_name"].as<std::string>().c_str());
  return bevfusion::create_core(param);
}

};  // namespace bevfusion

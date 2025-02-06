# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from .voxel_layer import dynamic_voxelize, hard_voxelize


class _Voxelization(Function):

    @staticmethod
    def forward(ctx,
                points,
                voxel_size,
                coors_range,
                max_points=35,
                max_voxels=20000,
                deterministic=True):
        """convert kitti points(N, >=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            dynamic_voxelize(points, coors, voxel_size, coors_range, 3)
            return coors
        else:
            voxels = points.new_zeros(
                size=(max_voxels, max_points, points.size(1)))
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(
                size=(max_voxels, ), dtype=torch.int)
            if len(points) == 0:
                voxel_num = 1
            else:
                voxel_num = hard_voxelize(
                    points,
                    voxels,
                    coors,
                    num_points_per_voxel,
                    voxel_size,
                    coors_range,
                    max_points,
                    max_voxels,
                    3,
                    deterministic,
                )
            # select the valid voxels
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply

class AdaptiveVoxelization(nn.Module):
    """Simple dynamic voxel feature encoder used in DV-SECOND.

    It simply averages the values of points in a voxel.
    But the number of points in a voxel is dynamic and varies.

    Args:
        voxel_size (tupe[float]): Size of a single voxel
        point_cloud_range (tuple[float]): Range of the point cloud and voxels
    """

    def __init__(
        self,
        point_cloud_range: list = [-50, -50, -1, 150, 50, 3],
        initial_voxel_size: tuple = (0.1, 0.1, 0.2),
        mode: str = "quadratic",
        scaling: float = 25.0,
        resolution: int = 1000,
        exp_scaling: float = 0.1,
    ):

        super(AdaptiveVoxelization, self).__init__()
        self.point_cloud_range = point_cloud_range
        self.initial_voxel_size = initial_voxel_size
        self.mode = mode
        self.scaling = scaling
        self.resolution = resolution
        self.exp_scaling = exp_scaling

    def calculate_voxel_centers(self):
        n_dim = 3
        voxel_centers = torch.stack(
            [
                f.flatten()
                for f in torch.meshgrid(
                    [
                        torch.linspace(
                            self.point_cloud_range[i],
                            self.point_cloud_range[i + n_dim],
                            self.resolution,
                        )
                        for i in range(n_dim)
                    ]
                )
            ],
            axis=0,
        )

        voxel_norms = torch.norm(voxel_centers, dim=0)
        normalization_constant = torch.max(torch.abs(voxel_norms))
        if self.mode == "linear":
            scaling_function = torch.abs(voxel_norms) / normalization_constant
        elif self.mode == "exponential":
            exp_scaling = self.exp_scaling
            scaling_function = torch.exp(torch.abs(voxel_norms) * exp_scaling) / torch.exp(
                normalization_constant * exp_scaling
            )
        elif self.mode == "quadratic":
            scaling_function = torch.abs(voxel_norms) ** 2 / normalization_constant**2
        else:
            raise NotImplementedError()
        voxel_centers *= (
            torch.ones_like(voxel_centers) + scaling_function * self.scaling
        )

        voxel_centers = voxel_centers[
            :,
            (voxel_centers[0] >= self.point_cloud_range[0])
            & (voxel_centers[0] <= self.point_cloud_range[3])
            & (voxel_centers[1] >= self.point_cloud_range[1])
            & (voxel_centers[1] <= self.point_cloud_range[4])
            & (voxel_centers[2] >= self.point_cloud_range[2])
            & (voxel_centers[2] <= self.point_cloud_range[5]),
        ]
        return voxel_centers

    def forward(self, pts):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        coors = self.calculate_voxel_centers()
        return dynamic_voxelize(pts, coors, self.initial_voxel_size, self.point_cloud_range, 3)


class Voxelization(nn.Module):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000,
                 deterministic=True):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w] removed
        self.pcd_shape = [*input_feat_shape, 1]  # [::-1]

    def forward(self, input):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return voxelization(
            input,
            self.voxel_size,
            self.point_cloud_range,
            self.max_num_points,
            max_voxels,
            self.deterministic,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ', deterministic=' + str(self.deterministic)
        tmpstr += ')'
        return tmpstr

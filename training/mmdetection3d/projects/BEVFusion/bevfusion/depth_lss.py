# modify from https://github.com/mit-han-lab/bevfusion
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.functional import adaptive_avg_pool2d

from mmdet3d.registry import MODELS
from .ops import bev_pool


def project_pixels_to_ground(cur_cam_intrinsic, cur_img_aug_matrix, cur_lidar_aug_matrix, cur_lidar2image, image_size, ground_height):
    # points_2d <-cur_img_aug_matrix- <-cur_lidar2image- -cur_lidar_aug_matrix-> points_3d
    xs = torch.arange(image_size[1], dtype=torch.float)
    ys = torch.arange(image_size[0], dtype=torch.float)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    points = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3).transpose(1, 0)  # 3, H*W
    points = points.to(device=cur_img_aug_matrix.device)
    
    # inverse imgaug
    points = points - cur_img_aug_matrix[:3, 3].reshape(3, 1)
    points = torch.inverse(cur_img_aug_matrix[:3, :3]).matmul(points)

    # inverse lidar2image
    points = points - cur_lidar2image[:3, 3].reshape(3, 1)
    points = torch.inverse(cur_lidar2image[:3, :3]).matmul(points)
    valid = points[2] < -1e-6
    scale = ground_height / points[2]
    points = points * scale

    # lidar aug
    points = cur_lidar_aug_matrix[:3, :3].matmul(points)
    points += cur_lidar_aug_matrix[:3, 3].reshape(3, 1)

    points = points.transpose(1, 0).reshape(image_size[0], image_size[1], 3)
    valid = valid.reshape(image_size[0], image_size[1])
    return valid, points

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                           for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class BaseViewTransform(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        dx.requires_grad = False
        bx.requires_grad = False
        nx.requires_grad = False
        self.register_buffer('dx', dx, persistent=False)
        self.register_buffer('bx', bx, persistent=False)
        self.register_buffer('nx', nx, persistent=False)

        self.C = out_channels
        frustum = self.create_frustum()
        self.register_buffer('frustum', frustum, persistent=False)
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound,
                         dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW))
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW,
                           dtype=torch.float).view(1, 1, fW).expand(D, fH, fW))
        ys = (
            torch.linspace(0, iH - 1, fH,
                           dtype=torch.float).view(1, fH, 1).expand(D, fH, fW))

        frustum = torch.stack((xs, ys, ds), -1)
        frustum.requires_grad = False
        return frustum

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots).view(B, N, 1, 1, 1, 3,
                                          3).matmul(points.unsqueeze(-1)))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if 'extra_rots' in kwargs:
            extra_rots = kwargs['extra_rots']
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3,
                                3).repeat(1, N, 1, 1, 1, 1, 1).matmul(
                                    points.unsqueeze(-1)).squeeze(-1))
        if 'extra_trans' in kwargs:
            extra_trans = kwargs['extra_trans']
            points += extra_trans.view(B, 1, 1, 1, 1,
                                       3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) /
                      self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = ((geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(
        self,
        img,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


@MODELS.register_module()
class LSSTransform(BaseViewTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x


class BaseDepthTransform(BaseViewTransform):

    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1,
                            *self.image_size).to(points[0].device)

        with torch.no_grad():
            for b in range(batch_size):
                # ts = (points[b][:, 3]).abs() < 1e-5  # select points with no time shift
                # cur_coords = points[b][ts, :3]
                cur_coords = points[b][:, :3]
                cur_img_aug_matrix = img_aug_matrix[b]
                cur_lidar_aug_matrix = lidar_aug_matrix[b]
                cur_lidar2image = lidar2image[b]

                # inverse aug
                cur_coords = cur_coords - cur_lidar_aug_matrix[:3, 3]
                cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                    cur_coords.transpose(1, 0))
                # lidar2image
                cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
                # get 2d coords
                dist = cur_coords[:, 2, :]
                cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
                cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

                # imgaug
                cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
                cur_coords = cur_coords[:, :2, :].transpose(1, 2)

                # normalize coords for grid sample
                cur_coords = cur_coords[..., [1, 0]]

                on_img = ((cur_coords[..., 0] < self.image_size[0])
                        & (cur_coords[..., 0] >= 0)
                        & (cur_coords[..., 1] < self.image_size[1])
                        & (cur_coords[..., 1] >= 0))
                
                for c in range(on_img.shape[0]):  # for each camera
                    masked_coords = cur_coords[c, on_img[c]].long()
                    masked_dist = dist[c, on_img[c]]
                    depth = depth.to(masked_dist.dtype)
                    depth[b, c, 0, masked_coords[:, 0],
                        masked_coords[:, 1]] = masked_dist

            extra_rots = lidar_aug_matrix[..., :3, :3]
            extra_trans = lidar_aug_matrix[..., :3, 3]
            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                intrins,
                post_rots,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x


@MODELS.register_module()
class DepthLSSTransform(BaseDepthTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample in (2, 4), downsample
            downsample_module = []
            for _ in range(downsample // 2):
                downsample_module.append(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False))
                downsample_module.append(nn.BatchNorm2d(out_channels))
                downsample_module.append(nn.ReLU(True))

            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                *downsample_module,
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, :self.D].softmax(dim=1).to(x.dtype)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x



@MODELS.register_module()
class DepthNLSSTransform(BaseDepthTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        depth_norm_loc: float = 20.0,
        depth_norm_scale: float = 10.0,
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depth_norm_loc = depth_norm_loc
        self.depth_norm_scale = depth_norm_scale
        self.dtransform = nn.Sequential(
            nn.Conv2d(2, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, 2 + self.C, 1),
        )
        if downsample > 1:
            assert downsample in (2, 4), downsample
            downsample_module = []
            for _ in range(downsample // 2):
                downsample_module.append(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False))
                downsample_module.append(nn.BatchNorm2d(out_channels))
                downsample_module.append(nn.ReLU(True))

            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                *downsample_module,
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        fd = self.dtransform(d)
        x = torch.cat([fd, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, :2].sigmoid().to(x.dtype)  # B*N, 2, H, W
        d_ref = adaptive_avg_pool2d(d[:, 1:2], depth.shape[-2:])  # B*N, 1, H, W
        loc = depth[:, 0] * self.depth_norm_loc - self.depth_norm_loc * 0.5 + d_ref[:, 0]  # B*N, H, W
        loc = loc.clamp(self.dbound[0], self.dbound[1])
        scale = depth[:, 1] * self.depth_norm_scale  # B*N, H, W
        scale = scale.clamp(1e-6, self.dbound[1] - self.dbound[0])
        dist = Normal(loc, scale)
        ds = torch.arange(*self.dbound, dtype=x.dtype).to(x.device).view(-1, 1, 1, 1)  # D, 1, 1, 1
        depth = dist.log_prob(ds).exp()  # D, B*N, H, W
        depth = depth.permute(1, 0, 2, 3).contiguous()  # B*N, D, H, W
        x = depth.unsqueeze(1) * x[:, 2:(2 + self.C)].unsqueeze(2)  # B*N, C, D, H, W

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)  # B, N, D, H, W, C
        return x

    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 2,
                            *self.image_size).to(points[0].device)

        with torch.no_grad():
            for b in range(batch_size):
                # ts = (points[b][:, 3]).abs() < 1e-5  # select points with no time shift
                # cur_coords = points[b][ts, :3]
                cur_coords = points[b][:, :3]
                ground_height = -metas[b]['lidar2ego'][2][3]  # TODO ego is not on ground
                cur_cam_intrinsic = cam_intrinsic[b]
                cur_img_aug_matrix = img_aug_matrix[b]
                cur_lidar_aug_matrix = lidar_aug_matrix[b]
                cur_lidar2image = lidar2image[b]

                # inverse aug
                cur_coords = cur_coords - cur_lidar_aug_matrix[:3, 3]
                cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                    cur_coords.transpose(1, 0))
                # lidar2image
                cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
                # get 2d coords
                dist = cur_coords[:, 2, :]
                cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
                cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

                # imgaug
                cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
                cur_coords = cur_coords[:, :2, :].transpose(1, 2)

                # normalize coords for grid sample
                cur_coords = cur_coords[..., [1, 0]]

                on_img = ((cur_coords[..., 0] < self.image_size[0])
                        & (cur_coords[..., 0] >= 0)
                        & (cur_coords[..., 1] < self.image_size[1])
                        & (cur_coords[..., 1] >= 0))
                
                for c in range(on_img.shape[0]):  # for each camera
                    masked_coords = cur_coords[c, on_img[c]].long()
                    masked_dist = dist[c, on_img[c]]
                    depth = depth.to(masked_dist.dtype)
                    depth[b, c, 0, masked_coords[:, 0],
                        masked_coords[:, 1]] = masked_dist
                    masked_coords, ground_points = project_pixels_to_ground(cur_cam_intrinsic[c], cur_img_aug_matrix[c], cur_lidar_aug_matrix, cur_lidar2image[c], self.image_size, ground_height)
                    d_ground = torch.sqrt(ground_points[..., 0] ** 2 + ground_points[..., 1] ** 2).to(depth)
                    d_ground = d_ground.clamp(self.dbound[0], self.dbound[1])
                    depth[b, c, 1, masked_coords] = d_ground[masked_coords]

            extra_rots = lidar_aug_matrix[..., :3, :3]
            extra_trans = lidar_aug_matrix[..., :3, 3]
            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                intrins,
                post_rots,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)

        x = self.downsample(x)
        return x


@MODELS.register_module()
class GroundLSSTransform(BaseDepthTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        depth_norm_loc: float = 20.0,
        depth_norm_scale: float = 10.0,
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depth_norm_loc = depth_norm_loc
        self.depth_norm_scale = depth_norm_scale
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, 2 + self.C, 1),
        )
        if downsample > 1:
            assert downsample in (2, 4), downsample
            downsample_module = []
            for _ in range(downsample // 2):
                downsample_module.append(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False))
                downsample_module.append(nn.BatchNorm2d(out_channels))
                downsample_module.append(nn.ReLU(True))

            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                *downsample_module,
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        fd = self.dtransform(d)
        x = torch.cat([fd, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, :2].sigmoid().to(x.dtype)  # B*N, 2, H, W
        d_ref = adaptive_avg_pool2d(d[:, 0:1], depth.shape[-2:])  # B*N, 1, H, W
        loc = depth[:, 0] * self.depth_norm_loc - self.depth_norm_loc * 0.5 + d_ref[:, 0]  # B*N, H, W
        loc = loc.clamp(self.dbound[0], self.dbound[1])
        scale = depth[:, 1] * self.depth_norm_scale  # B*N, H, W
        scale = scale.clamp(1e-6, self.dbound[1] - self.dbound[0])
        dist = Normal(loc, scale)
        ds = torch.arange(*self.dbound, dtype=x.dtype).to(x.device).view(-1, 1, 1, 1)  # D, 1, 1, 1
        depth = dist.log_prob(ds).exp()  # D, B*N, H, W
        depth = depth.permute(1, 0, 2, 3).contiguous()  # B*N, D, H, W
        x = depth.unsqueeze(1) * x[:, 2:(2 + self.C)].unsqueeze(2)  # B*N, C, D, H, W

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)  # B, N, D, H, W, C
        return x

    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1,
                            *self.image_size).to(points[0].device)

        with torch.no_grad():
            for b in range(batch_size):
                ground_height = -metas[b]['lidar2ego'][2][3]  # TODO ego is not on ground
                cur_cam_intrinsic = cam_intrinsic[b]
                cur_img_aug_matrix = img_aug_matrix[b]
                cur_lidar_aug_matrix = lidar_aug_matrix[b]
                cur_lidar2image = lidar2image[b]

                for c in range(len(cur_cam_intrinsic)):  # for each camera
                    masked_coords, ground_points = project_pixels_to_ground(cur_cam_intrinsic[c], cur_img_aug_matrix[c], cur_lidar_aug_matrix, cur_lidar2image[c], self.image_size, ground_height)
                    d_ground = torch.sqrt(ground_points[..., 0] ** 2 + ground_points[..., 1] ** 2).to(depth)
                    d_ground = d_ground.clamp(self.dbound[0], self.dbound[1])
                    depth[b, c, 0, masked_coords] = d_ground[masked_coords]

            extra_rots = lidar_aug_matrix[..., :3, :3]
            extra_trans = lidar_aug_matrix[..., :3, 3]
            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                intrins,
                post_rots,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)

        x = self.downsample(x)
        return x

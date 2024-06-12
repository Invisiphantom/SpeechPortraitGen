import os
import torch
import torch.nn.functional as F
import librosa
import numpy as np
import importlib
import tqdm
import copy
import cv2
from scipy.spatial.transform import Rotation


def load_img_to_512_hwc_array(img_name):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    return img


def load_img_to_normalized_512_bchw_tensor(img_name):
    img = load_img_to_512_hwc_array(img_name)
    img = ((torch.tensor(img) - 127.5) / 127.5).float().unsqueeze(0).permute(0, 3, 1, 2)  # [b,c,h,w]
    return img


def mirror_index(index, len_seq):
    """
    get mirror index when indexing a sequence and the index is larger than len_pose
    args:
        index: int
        len_pose: int
    return:
        mirror_index: int
    """
    turn = index // len_seq
    res = index % len_seq
    if turn % 2 == 0:
        return res  # forward indexing
    else:
        return len_seq - res - 1  # reverse indexing


def smooth_camera_sequence(camera, kernel_size=7):
    """
    smooth the camera trajectory (i.e., rotation & translation)...
    args:
        camera: [N, 25] or [N, 16]. np.ndarray
        kernel_size: int
    return:
        smoothed_camera: [N, 25] or [N, 16]. np.ndarray
    """
    # poses: [N, 25], numpy array
    N = camera.shape[0]
    K = kernel_size // 2
    poses = camera[:, :16].reshape([-1, 4, 4]).copy()
    trans = poses[:, :3, 3].copy()  # [N, 3]
    rots = poses[:, :3, :3].copy()  # [N, 3, 3]

    for i in range(N):
        start = max(0, i - K)
        end = min(N, i + K + 1)
        poses[i, :3, 3] = trans[start:end].mean(0)
        try:
            poses[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()
        except:
            if i == 0:
                poses[i, :3, :3] = rots[i]
            else:
                poses[i, :3, :3] = poses[i - 1, :3, :3]
    poses = poses.reshape([-1, 16])
    camera[:, :16] = poses
    return camera


def smooth_features_xd(in_tensor, kernel_size=7):
    """
    smooth the feature maps
    args:
        in_tensor: [T, c,h,w] or [T, c1,c2,h,w]
        kernel_size: int
    return:
        out_tensor: [T, c,h,w] or [T, c1,c2,h,w]
    """
    t = in_tensor.shape[0]
    ndim = in_tensor.ndim
    pad = (kernel_size - 1) // 2
    in_tensor = torch.cat(
        [torch.flip(in_tensor[0:pad], dims=[0]), in_tensor, torch.flip(in_tensor[t - pad : t], dims=[0])], dim=0
    )
    if ndim == 2:  # tc
        _, c = in_tensor.shape
        in_tensor = in_tensor.permute(1, 0).reshape([-1, 1, t + 2 * pad])  # [c, 1, t]
    elif ndim == 4:  # tchw
        _, c, h, w = in_tensor.shape
        in_tensor = in_tensor.permute(1, 2, 3, 0).reshape([-1, 1, t + 2 * pad])  # [c, 1, t]
    elif ndim == 5:  # tcchw, like deformation
        _, c1, c2, h, w = in_tensor.shape
        in_tensor = in_tensor.permute(1, 2, 3, 4, 0).reshape([-1, 1, t + 2 * pad])  # [c, 1, t]
    else:
        raise NotImplementedError()
    avg_kernel = (
        1 / kernel_size * torch.Tensor([1.0] * kernel_size).reshape([1, 1, kernel_size]).float().to(in_tensor.device)
    )  # [1, 1, kw]
    out_tensor = F.conv1d(in_tensor, avg_kernel)
    if ndim == 2:  # tc
        return out_tensor.reshape([c, t]).permute(1, 0)
    elif ndim == 4:  # tchw
        return out_tensor.reshape([c, h, w, t]).permute(3, 0, 1, 2)
    elif ndim == 5:  # tcchw, like deformation
        return out_tensor.reshape([c1, c2, h, w, t]).permute(4, 0, 1, 2, 3)



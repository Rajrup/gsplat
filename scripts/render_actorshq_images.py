import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v3 as imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import tqdm

from gsplat.rendering import rasterization

FORCE_OPENCV_MODEL = True

@dataclass
class RenderConfig:
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    antialiased: bool = False
    camera_model: str = "pinhole"
    with_ut: bool = False
    with_eval3d: bool = False
    data_factor: int = 1
    test_every: int = 8
    normalize_world_space: bool = True

def _load_splats(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    if "splats" not in ckpt:
        raise KeyError(f"Missing splats in checkpoint: {ckpt_path}")
    splats = ckpt["splats"]
    means = splats["means"].to(device)
    quats = F.normalize(splats["quats"].to(device), p=2, dim=-1)
    scales = torch.exp(splats["scales"].to(device))
    opacities = torch.sigmoid(splats["opacities"].to(device))
    if "sh0" not in splats or "shN" not in splats:
        raise KeyError("Checkpoint missing sh0/shN; expected SH colors.")
    sh0 = splats["sh0"].to(device)
    shN = splats["shN"].to(device)
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    return means, quats, scales, opacities, colors, sh_degree


def _get_cam_params(cam):
    params = getattr(cam, "params", None)
    if params is not None:
        return list(params)
    k1 = getattr(cam, "k1", 0.0)
    k2 = getattr(cam, "k2", 0.0)
    p1 = getattr(cam, "p1", 0.0)
    p2 = getattr(cam, "p2", 0.0)
    k3 = getattr(cam, "k3", 0.0)
    k4 = getattr(cam, "k4", 0.0)
    return [k1, k2, p1, p2, k3, k4]


def _get_rel_paths(path_dir: str):
    paths = []
    for dp, _, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)
    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm.tqdm(image_files, desc="downscale"):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


def _similarity_from_cameras(c2w, strict_scaling=False):
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)
    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
    translate = -np.median(nearest, axis=0)
    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale
    return transform


def _align_principal_axes(point_cloud):
    centroid = np.median(point_cloud, axis=0)
    translated = point_cloud - centroid
    covariance_matrix = np.cov(translated, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rotation_matrix = eigenvectors.T
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid
    return transform


def _transform_points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _transform_cameras(matrix, camtoworlds):
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


def _normalize_cameras_and_points(camtoworlds, points):
    T1 = _similarity_from_cameras(camtoworlds)
    camtoworlds = _transform_cameras(T1, camtoworlds)
    points = _transform_points(T1, points)
    T2 = _align_principal_axes(points)
    camtoworlds = _transform_cameras(T2, camtoworlds)
    points = _transform_points(T2, points)
    if np.median(points[:, 2]) > np.mean(points[:, 2]):
        t3 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        camtoworlds = _transform_cameras(t3, camtoworlds)
        points = _transform_points(t3, points)
    return camtoworlds


def _load_colmap_scene(data_dir: str, cfg: RenderConfig):
    try:
        from pycolmap import SceneManager
    except ImportError as exc:
        raise ImportError("pycolmap is required for COLMAP parsing.") from exc

    colmap_dir = os.path.join(data_dir, "sparse/0/")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(data_dir, "sparse")
    if not os.path.exists(colmap_dir):
        raise FileNotFoundError(f"COLMAP directory not found: {colmap_dir}")

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()

    imdata = manager.images
    if len(imdata) == 0:
        raise ValueError("No images found in COLMAP.")

    params_dict = {}
    imsize_dict = {}
    mask_dict = {}
    camtype = "opencv"
    w2c_mats = []
    camera_ids = []
    Ks_dict = {}
    image_names = []
    for image_id in imdata:
        im = imdata[image_id]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), np.array([[0, 0, 0, 1]])], axis=0)
        w2c_mats.append(w2c)
        camera_id = im.camera_id
        camera_ids.append(camera_id)
        cam = manager.cameras[camera_id]
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K[:2, :] /= cfg.data_factor
        Ks_dict[camera_id] = K
        imsize_dict[camera_id] = (cam.width // cfg.data_factor, cam.height // cfg.data_factor)
        image_names.append(im.name)

        model = "OPENCV" if FORCE_OPENCV_MODEL else getattr(cam, "model", None)
        if model in (None, "PINHOLE", "SIMPLE_PINHOLE"):
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        elif model == "SIMPLE_RADIAL":
            cam_params = _get_cam_params(cam)
            params = np.array([cam_params[0], 0.0, 0.0, 0.0], dtype=np.float32)
            camtype = "perspective"
        elif model == "RADIAL":
            cam_params = _get_cam_params(cam)
            params = np.array([cam_params[0], cam_params[1], 0.0, 0.0], dtype=np.float32)
            camtype = "perspective"
        elif model == "OPENCV":
            cam_params = _get_cam_params(cam)
            params = np.array([cam_params[0], cam_params[1], cam_params[2], cam_params[3]], dtype=np.float32)
            camtype = "perspective"
        elif model == "OPENCV_FISHEYE":
            cam_params = _get_cam_params(cam)
            params = np.array([cam_params[0], cam_params[1], cam_params[4], cam_params[5]], dtype=np.float32)
            camtype = "fisheye"
        else:
            raise ValueError(f"Unsupported camera model: {model}")
        params_dict[camera_id] = params
        mask_dict[camera_id] = None

    w2c_mats = np.stack(w2c_mats, axis=0)
    camtoworlds = np.linalg.inv(w2c_mats)

    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]
    camera_ids = [camera_ids[i] for i in inds]

    if cfg.data_factor > 1:
        image_dir_suffix = f"_{cfg.data_factor}"
    else:
        image_dir_suffix = ""
    colmap_image_dir = os.path.join(data_dir, "images")
    image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
    if not os.path.isdir(image_dir):
        image_dir = colmap_image_dir

    colmap_files = sorted(_get_rel_paths(colmap_image_dir))
    image_files = sorted(_get_rel_paths(image_dir))
    if cfg.data_factor > 1 and image_files and os.path.splitext(image_files[0])[1].lower() == ".jpg":
        image_dir = _resize_image_folder(
            colmap_image_dir, image_dir + "_png", factor=cfg.data_factor
        )
        image_files = sorted(_get_rel_paths(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

    points = manager.points3D.astype(np.float32) if manager.points3D is not None else None
    if cfg.normalize_world_space and points is not None and len(points) > 0:
        camtoworlds = _normalize_cameras_and_points(camtoworlds, points)

    actual_image = imageio.imread(image_paths[0])[..., :3]
    actual_height, actual_width = actual_image.shape[:2]
    colmap_width, colmap_height = imsize_dict[camera_ids[0]]
    s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
    for camera_id, K in Ks_dict.items():
        K[0, :] *= s_width
        K[1, :] *= s_height
        Ks_dict[camera_id] = K
        width, height = imsize_dict[camera_id]
        imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

    mapx_dict = {}
    mapy_dict = {}
    roi_undist_dict = {}
    for camera_id, params in params_dict.items():
        if len(params) == 0:
            continue
        K = Ks_dict[camera_id]
        width, height = imsize_dict[camera_id]
        if camtype == "perspective":
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            mask = None
        else:
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
                indexing="xy",
            )
            x1 = (grid_x - cx) / fx
            y1 = (grid_y - cy) / fy
            theta = np.sqrt(x1**2 + y1**2)
            r = (
                1.0
                + params[0] * theta**2
                + params[1] * theta**4
                + params[2] * theta**6
                + params[3] * theta**8
            )
            mapx = (fx * x1 * r + width // 2).astype(np.float32)
            mapy = (fy * y1 * r + height // 2).astype(np.float32)
            mask = np.logical_and(
                np.logical_and(mapx > 0, mapx < width),
                np.logical_and(mapy > 0, mapy < height),
            )
            roi_undist = (0, 0, width, height)
        mapx_dict[camera_id] = mapx
        mapy_dict[camera_id] = mapy
        roi_undist_dict[camera_id] = roi_undist
        mask_dict[camera_id] = mask

    return camtoworlds, camera_ids, Ks_dict, image_paths, params_dict, mapx_dict, mapy_dict, roi_undist_dict


def _select_indices(num_images: int, split: str, test_every: int):
    if split == "all":
        return list(range(num_images))
    if test_every <= 0:
        return list(range(num_images))
    if split == "val":
        return [i for i in range(num_images) if i % test_every == 0]
    return [i for i in range(num_images) if i % test_every != 0]


def _render_split(
    split: str,
    camtoworlds: np.ndarray,
    camera_ids,
    Ks_dict,
    image_paths,
    params_dict,
    mapx_dict,
    mapy_dict,
    roi_undist_dict,
    device: torch.device,
    output_dir: Path,
    cfg: RenderConfig,
    splats,
    sh_degree: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    means, quats, scales, opacities, colors = splats
    rasterize_mode = "antialiased" if cfg.antialiased else "classic"

    indices = _select_indices(len(image_paths), split, cfg.test_every)
    for out_idx, i in enumerate(tqdm.tqdm(indices, desc=f"render-{split}")):
        image = imageio.imread(image_paths[i])[..., :3]
        camera_id = camera_ids[i]
        params = params_dict[camera_id]
        if len(params) > 0:
            mapx = mapx_dict[camera_id]
            mapy = mapy_dict[camera_id]
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]
        height, width = image.shape[:2]
        camtoworld = torch.from_numpy(camtoworlds[i]).float().to(device).unsqueeze(0)
        K = torch.from_numpy(Ks_dict[camera_id]).float().to(device).unsqueeze(0)

        render_colors, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworld),
            Ks=K,
            width=width,
            height=height,
            packed=cfg.packed,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            distributed=False,
            camera_model=cfg.camera_model,
            with_ut=cfg.with_ut,
            with_eval3d=cfg.with_eval3d,
            sh_degree=sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
        )

        rgb = render_colors[0].clamp(0.0, 1.0).cpu().numpy()
        rgb = (rgb * 255.0).astype(np.uint8)
        imageio.imwrite(output_dir / f"{split}_{out_idx:04d}.png", rgb)


def _build_data_dir(args: argparse.Namespace) -> str:
    if args.data_dir is not None:
        return args.data_dir
    frame_id = 0 if args.frame_id is None else args.frame_id
    resolution = 4 if args.resolution is None else args.resolution
    actorshq_data_dir = args.actorshq_data_dir
    if actorshq_data_dir is None:
        actorshq_data_dir = "/ssd1/rajrup/Project/gsplat/data/Actor01/Sequence1/"
    return os.path.join(actorshq_data_dir, str(frame_id), f"resolution_{resolution}")


def main():
    parser = argparse.ArgumentParser(
        description="Render ActorHQ images from a gsplat checkpoint (standalone)."
    )
    parser.add_argument(
        "--ckpt",
        default=(
            "/ssd1/rajrup/Project/gsplat/results/"
            "actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/"
            "resolution_4/0/ckpts/ckpt_29999_rank0.pt"
        ),
    )
    parser.add_argument("--actorshq-data-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--antialiased", action="store_true")
    parser.add_argument("--camera-model", default="pinhole")
    parser.add_argument("--with-ut", action="store_true")
    parser.add_argument("--with-eval3d", action="store_true")
    parser.add_argument("--data-factor", type=int, default=None)
    parser.add_argument("--test-every", type=int, default=1)
    parser.add_argument("--normalize-world-space", action="store_true", default=True)
    parser.add_argument("--no-normalize-world-space", dest="normalize_world_space", action="store_false")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    data_dir = _build_data_dir(args)
    if args.data_factor is None:
        if os.path.isdir(os.path.join(data_dir, "images")):
            data_factor = 4
        else:
            data_factor = 1
    else:
        data_factor = args.data_factor

    cfg = RenderConfig(
        near_plane=args.near_plane,
        far_plane=args.far_plane,
        packed=args.packed,
        antialiased=args.antialiased,
        camera_model=args.camera_model,
        with_ut=args.with_ut,
        with_eval3d=args.with_eval3d,
        data_factor=data_factor,
        test_every=args.test_every,
        normalize_world_space=args.normalize_world_space,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    means, quats, scales, opacities, colors, sh_degree = _load_splats(
        ckpt_path, device
    )
    splats = (means, quats, scales, opacities, colors)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else Path("rendered_images")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        camtoworlds,
        camera_ids,
        Ks_dict,
        image_paths,
        params_dict,
        mapx_dict,
        mapy_dict,
        roi_undist_dict,
    ) = _load_colmap_scene(data_dir, cfg)

    if args.split in ("train", "all"):
        _render_split(
            "train",
            camtoworlds,
            camera_ids,
            Ks_dict,
            image_paths,
            params_dict,
            mapx_dict,
            mapy_dict,
            roi_undist_dict,
            device,
            output_dir / "train",
            cfg,
            splats,
            sh_degree,
        )
    if args.split in ("val", "all"):
        _render_split(
            "val",
            camtoworlds,
            camera_ids,
            Ks_dict,
            image_paths,
            params_dict,
            mapx_dict,
            mapy_dict,
            roi_undist_dict,
            device,
            output_dir / "val",
            cfg,
            splats,
            sh_degree,
        )


if __name__ == "__main__":
    main()

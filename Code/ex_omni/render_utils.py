import torch
import numpy as np
import os
import shutil
import subprocess
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    DirectionalLights,
    TexturesVertex,
    Materials,
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.camera_utils import join_cameras_as_batch


def _resolve_ffmpeg_binary():
    """Locate an ffmpeg executable with libx264 support.

    Resolution order (first hit wins):
      1. ``$EX_OMNI_FFMPEG`` — explicit override for a compatible static
         build.
      2. ``imageio_ffmpeg.get_ffmpeg_exe()`` — statically built binary that
         always ships libx264, avoiding system builds compiled without it
         (e.g. an ffmpeg build with ``--enable-gpl`` but no
         ``--enable-libx264``).
      3. ``shutil.which("ffmpeg")`` — whatever is on ``$PATH``.
    """
    override = os.environ.get("EX_OMNI_FFMPEG")
    if override and os.path.isfile(override):
        return override

    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg") or "ffmpeg"


_FFMPEG_BIN = _resolve_ffmpeg_binary()


# Model blendshape 顺序
MODEL_BS_LIST = [
    "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeWideLeft", "eyeWideRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "browDownLeft", "browDownRight",
    "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "jawOpen",
    "mouthClose", "jawLeft", "jawRight", "jawForward", "mouthUpperUpLeft",
    "mouthUpperUpRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthRollUpper", "mouthRollLower", "mouthSmileLeft", "mouthSmileRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthFrownLeft", "mouthFrownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthFunnel", "mouthLeft", "mouthRight",
    "mouthShrugLower", "mouthShrugUpper", "noseSneerLeft", "noseSneerRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight"
]

Standard_MODEL_BS_LIST = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]

def create_blendshape_mapping(TEMPLATE_BS_LIST, model_bs_list):
    """
    创建从模型权重顺序到网格 blendshape 顺序的映射
    Args:
        TEMPLATE_BS_LIST: 网格文件中的 blendshape 名称列表
        model_bs_list: 模型输出的 blendshape 名称列表
    Returns:
        mapping_indices: (52,) 映射索引数组，mapping_indices[i] 表示 model_bs_list[i] 
                        对应 TEMPLATE_BS_LIST 中的索引位置
    """
    # 将 numpy bytes 转换为字符串（如果需要）
    if isinstance(TEMPLATE_BS_LIST[0], bytes):
        TEMPLATE_BS_LIST = [name.decode('utf-8') for name in TEMPLATE_BS_LIST]
    
    # 创建网格名称到索引的映射
    gt_name_to_idx = {name: idx for idx, name in enumerate(TEMPLATE_BS_LIST)}
    
    # 创建映射索引
    mapping_indices = []
    
    for model_name in model_bs_list:
        if model_name in gt_name_to_idx:
            mapping_indices.append(gt_name_to_idx[model_name])
        else:
            mapping_indices.append(-1)  # 使用 -1 标记缺失
    
    return np.array(mapping_indices)

def remap_weights(weights, mapping_indices):
    T, num_bs = weights.shape
    remapped = np.zeros((T, num_bs), dtype=weights.dtype)
    
    for model_idx, gt_idx in enumerate(mapping_indices):
        if gt_idx != -1:  # 跳过缺失的 blendshape
            remapped[:, gt_idx] = weights[:, model_idx]
    
    return remapped

def load_mesh_data(template_path):
    data = np.load(template_path)
    
    meanshape = torch.from_numpy(data['meanshape']).float()
    faces = torch.from_numpy(data['faces']).long()

    # faces可能是1-indexed，需要转换为0-indexed
    if faces.min() == 1:
        faces = faces - 1
    
    # blendshape已经是差值形式
    blendshape = torch.from_numpy(data['blendshape']).float()
    TEMPLATE_BS_LIST = data['bs_names']
    
    return meanshape, faces, blendshape, TEMPLATE_BS_LIST

def apply_blendshapes(meanshape, blendshape, weights):
    weights = weights.reshape(52, 1, 1)  # (52, 1, 1)
    weighted_deltas = (blendshape * weights).sum(dim=0)  # (V, 3)
    return weighted_deltas + meanshape

def apply_blendshapes_batch(meanshape, blendshape, weights):
    if weights.ndim == 1:
        weights = weights.unsqueeze(0)
    weighted_deltas = torch.einsum("tb,bvc->tvc", weights, blendshape)
    return weighted_deltas + meanshape.unsqueeze(0)

def setup_renderer(meanshape, image_size=(512, 512)):
    # 计算网格边界
    x_max = y_max = meanshape[..., 0:2].abs().max()
    
    # 设置旋转矩阵（镜像变换）
    R = torch.eye(3, device=meanshape.device)
    R[0, 0] = -1  # X轴镜像
    R[2, 2] = -1  # Z轴镜像
    R = R.unsqueeze(0)
    
    # 相机位置
    T = torch.zeros(3, device=meanshape.device)
    T[2] = 10  # 相机后退
    T = T.unsqueeze(0)
    
    # 正交相机
    cameras = FoVOrthographicCameras(
        device=meanshape.device,
        R=R,
        T=T,
        znear=0.01,
        zfar=3,
        max_x=x_max * 1.2,
        min_x=-x_max * 1.2,
        max_y=y_max * 1.2,
        min_y=-y_max * 1.2,
    )
    
    # 光栅化设置
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    # 方向光
    lights = DirectionalLights(
        device=meanshape.device,
        direction=((0, 0, 1),),
        ambient_color=((0.3, 0.3, 0.3),),
        diffuse_color=((0.6, 0.6, 0.6),),
        specular_color=((0.1, 0.1, 0.1),)
    )
    
    # 材质
    materials = Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=15,
        device=meanshape.device
    )
    
    # 混合参数
    blend_params = BlendParams(
        sigma=0.0,
        gamma=0.0,
        background_color=(0.0, 0.0, 0.0)
    )
    
    # 着色器
    shader = HardPhongShader(
        device=meanshape.device,
        cameras=cameras,
        lights=lights,
        materials=materials,
        blend_params=blend_params
    )
    
    # 渲染器
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=shader
    )
    
    return renderer

def render_frames(renderer, verts, faces, vertex_color=None):
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)

    batch_size, num_verts = verts.shape[:2]
    faces_batch = faces.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    if vertex_color is None:
        verts_rgb = torch.ones_like(verts)
    else:
        verts_rgb = torch.as_tensor(
            vertex_color,
            dtype=verts.dtype,
            device=verts.device,
        ).view(1, 1, 3).expand(batch_size, num_verts, 3).contiguous()
    textures = TexturesVertex(verts_features=verts_rgb)

    meshes = Meshes(
        verts=verts,
        faces=faces_batch,
        textures=textures
    )

    cameras = renderer.rasterizer.cameras
    if batch_size > 1 and cameras.R.shape[0] == 1:
        cameras = join_cameras_as_batch([cameras] * batch_size)

    with torch.no_grad():
        images = renderer(meshes, cameras=cameras)

    return images[..., :3]

def render_frame(renderer, verts, faces):
    return render_frames(renderer, verts, faces)[0]


def images_to_video(image_array, output_path, fps=30, audio_input=None):
    T, H, W, C = image_array.shape
    assert C == 3, "Expected RGB images"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if H % 2 != 0:
        image_array = np.pad(image_array, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant')
        H += 1
    if W % 2 != 0:
        image_array = np.pad(image_array, ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant')
        W += 1

    image_array_bgr = image_array[..., ::-1]  # RGB -> BGR

    command = [
        _FFMPEG_BIN,
        '-y',  # 覆盖输出文件
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',  # 从 stdin 输入图像
    ]

    if audio_input is not None:
        command += ['-i', audio_input, '-map', '0:v', '-map', '1:a', '-c:a', 'aac']  # 自动截断到较短流
    else:
        command += ['-an']  # 无音频

    command += [
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        all_frames = image_array_bgr.tobytes()
        stdout, stderr = process.communicate(input=all_frames, timeout=600)
    except subprocess.TimeoutExpired:
        process.kill()
        stderr = process.stderr.read()
        raise RuntimeError(f"FFmpeg 超时\n{stderr.decode()}") from None
    except Exception as e:
        process.kill()
        raise RuntimeError(f"写入视频时发生错误: {e}") from None

    if process.returncode != 0:
        raise RuntimeError(
            f"FFmpeg 编码失败 (返回码 {process.returncode})\n"
            f"命令: {' '.join(command)}\n{stderr.decode()}"
        )


class ParallelFrameRenderer:
    """Render independent batches of one response across GPUs, preserving order."""

    def __init__(self, meanshape, faces, blendshape, devices, image_size=(512, 512)):
        from concurrent.futures import ThreadPoolExecutor

        self.devices = [torch.device(device) for device in devices]
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError("Render devices must be nonempty and distinct")
        for device in self.devices:
            if device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
                raise ValueError(f"Expected a visible explicit CUDA render device: {device}")
        self.closed = False
        self.executors = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"render-{device.index}")
                          for device in self.devices]
        assets = tuple(tensor.detach().cpu() for tensor in (meanshape, faces, blendshape))
        try:
            futures = [executor.submit(self._initialize, device, assets, image_size)
                       for executor, device in zip(self.executors, self.devices)]
            self.workers = [future.result() for future in futures]
            # cuSOLVER's lazy torch.inverse initialization is process-global and
            # must finish before multiple render threads enter camera inversion.
            for executor, worker in zip(self.executors, self.workers):
                executor.submit(self._render, worker, np.zeros((1, 52), dtype=np.float32)).result()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _initialize(device, assets, image_size):
        torch.cuda.set_device(device)
        mean, faces, shapes = [tensor.to(device) for tensor in assets]
        return device, mean, faces, shapes, setup_renderer(mean, image_size)

    @staticmethod
    @torch.inference_mode()
    def _render(worker, weights):
        device, mean, faces, shapes, renderer = worker
        torch.cuda.set_device(device)
        weights = torch.as_tensor(weights, device=device, dtype=mean.dtype)
        vertices = apply_blendshapes_batch(mean, shapes, weights)
        return render_frames(renderer, vertices, faces).clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()

    def render(self, weights, batch_size=16):
        from collections import deque

        if self.closed:
            raise RuntimeError("ParallelFrameRenderer is closed")
        if batch_size < 1 or len(weights.shape) != 2 or weights.shape[1] != 52:
            raise ValueError("Expected [frames, 52] weights and positive batch size")
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().float().cpu().numpy()
        weights = np.array(weights, dtype=np.float32, order="C", copy=True)
        pending = deque()
        try:
            for batch, start in enumerate(range(0, len(weights), batch_size)):
                index = batch % len(self.workers)
                pending.append(self.executors[index].submit(
                    self._render, self.workers[index], weights[start:start + batch_size]))
                if len(pending) >= 2 * len(self.workers):
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()
            for future in pending:
                if not future.cancelled():
                    future.result()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=True)
        self.workers = []


class VideoStreamWriter:
    """Feed rendered batches to ffmpeg while the next batch is rendered."""

    def __init__(self, output_path, image_size=(512, 512), fps=30, audio_input=None):
        import tempfile

        height, width = image_size
        if height % 2 or width % 2:
            raise ValueError("VideoStreamWriter requires even image dimensions")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        command = [
            _FFMPEG_BIN, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-',
        ]
        if audio_input is not None:
            command += ['-i', audio_input, '-map', '0:v', '-map', '1:a', '-c:a', 'aac']
        else:
            command += ['-an']
        command += ['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', output_path]
        self.image_size = image_size
        self.command = command
        self.stderr = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.stderr
        )
        self.closed = False

    def write(self, frames):
        if self.closed:
            raise RuntimeError("VideoStreamWriter is closed")
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (*self.image_size, 3):
            raise ValueError(f"Expected frames with shape (N, {self.image_size[0]}, {self.image_size[1]}, 3)")
        if frames.dtype != np.uint8:
            raise ValueError("VideoStreamWriter requires uint8 frames")
        try:
            self.process.stdin.write(np.ascontiguousarray(frames[..., ::-1]).tobytes())
        except BrokenPipeError as exc:
            self.process.wait()
            self.stderr.seek(0)
            raise RuntimeError(f"FFmpeg exited while writing frames: {self.stderr.read().decode(errors='replace')}") from exc

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.process.stdin.close()
        self.process.stdin = None
        try:
            self.process.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("FFmpeg timed out while finalizing video") from None
        finally:
            self.stderr.seek(0)
            error = self.stderr.read().decode(errors='replace')
            self.stderr.close()
        if self.process.returncode:
            raise RuntimeError(f"FFmpeg failed ({self.process.returncode}): {error}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.process.kill()
            self.process.wait()
            self.stderr.close()
            self.closed = True
        else:
            self.close()

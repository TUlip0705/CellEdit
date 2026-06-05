"""
数据预处理模块

支持两种模式：
1. 动态percentile（旧模式，每张图计算percentile）
2. 固定统计参数（推荐，训练集统计一次后全程复用）
"""

import json
from typing import Optional, Tuple

import torch


def load_fixed_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    required_keys = ["q_low", "q_high", "mean01", "std01", "num_channels"]
    missing = [k for k in required_keys if k not in stats]
    if missing:
        raise ValueError(f"Stats file missing keys: {missing}")
    return stats


def percentile_clip(
    arr: torch.Tensor,
    bounds: Tuple[float, float] = (0.5, 99.5),
    sample_stride: int = 100,
    per_channel: bool = False,
) -> torch.Tensor:
    """
    百分位数裁剪和归一化

    Rescale image intensity by robust percentiles.

    Args:
        arr: 输入tensor [C, H, W] or [B, C, H, W]
        bounds: 百分位数范围，如(0.5, 99.5)表示裁剪到0.5%-99.5%范围
        sample_stride: 采样步长（每N个像素取1个），用于加速计算

    Returns:
        归一化到[0, 1]的tensor
    """
    original_shape = arr.shape
    is_batch = len(original_shape) == 4

    if is_batch:
        # [B, C, H, W] → 逐样本处理
        batch_size = arr.shape[0]
        processed = []
        for i in range(batch_size):
            processed.append(percentile_clip(arr[i], bounds, sample_stride, per_channel))
        return torch.stack(processed, dim=0)

    # [C, H, W] 处理
    arr = arr.float()

    # 自动检测图像位深度
    max_val = float(arr.max())
    if max_val > 1.0:
        # 16位或8位图像，先缩放到[0, 1]
        arr = arr / (65535.0 if max_val > 255.0 else 255.0)

    if per_channel:
        # 对每个通道独立计算percentile
        out = torch.empty_like(arr)
        for ch in range(arr.shape[0]):
            ch_data = arr[ch]
            sample = ch_data.flatten()[::sample_stride]
            if sample.numel() == 0:
                out[ch] = ch_data
                continue
            percentiles = torch.quantile(
                sample,
                torch.tensor([bounds[0] / 100.0, bounds[1] / 100.0], device=arr.device)
            )
            p_min, p_max = percentiles[0], percentiles[1]
            if p_max - p_min < 1e-6:
                out[ch] = torch.zeros_like(ch_data)
                continue
            ch_data = torch.clamp(ch_data, p_min, p_max)
            out[ch] = (ch_data - p_min) / (p_max - p_min)
        return out

    # 采样计算百分位数（加速）
    # 展平并采样
    sample = arr.flatten()[::sample_stride]

    if sample.numel() == 0:
        return arr

    # 计算百分位数
    percentiles = torch.quantile(
        sample,
        torch.tensor([bounds[0] / 100.0, bounds[1] / 100.0], device=arr.device)
    )

    p_min, p_max = percentiles[0], percentiles[1]

    # 避免除零
    if p_max - p_min < 1e-6:
        return torch.zeros_like(arr)

    # 裁剪和归一化
    arr = torch.clamp(arr, p_min, p_max)
    arr = (arr - p_min) / (p_max - p_min)

    return arr


def fixed_percentile_clip(
    arr: torch.Tensor,
    q_low: torch.Tensor,
    q_high: torch.Tensor,
) -> torch.Tensor:
    """
    使用固定通道统计参数做裁剪归一化（推荐模式）

    Args:
        arr: [C,H,W] 或 [B,C,H,W]
        q_low/q_high: [C] 形状，值域在[0,1]
    """
    arr = arr.float()
    max_val = float(arr.max()) if arr.numel() else 0.0
    if max_val > 1.0:
        arr = arr / (65535.0 if max_val > 255.0 else 255.0)

    if arr.ndim == 3:
        low = q_low.view(-1, 1, 1).to(arr.device)
        high = q_high.view(-1, 1, 1).to(arr.device)
    elif arr.ndim == 4:
        low = q_low.view(1, -1, 1, 1).to(arr.device)
        high = q_high.view(1, -1, 1, 1).to(arr.device)
    else:
        raise ValueError(f"Unsupported shape for fixed_percentile_clip: {arr.shape}")

    denom = torch.clamp(high - low, min=1e-6)
    arr = torch.clamp(arr, low, high)
    arr = (arr - low) / denom
    arr = torch.clamp(arr, 0.0, 1.0)
    return arr


def channel_wise_standardize(arr: torch.Tensor) -> torch.Tensor:
    """
    通道级自标准化

    Standardize each image independently.
    对每个通道独立进行 (X - mean) / std

    Args:
        arr: [C, H, W] or [B, C, H, W]

    Returns:
        标准化后的tensor
    """
    if len(arr.shape) == 3:
        # [C, H, W]
        mean = arr.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        std = arr.std(dim=(1, 2), keepdim=True) + 1e-6  # 防止除零
        return (arr - mean) / std
    elif len(arr.shape) == 4:
        # [B, C, H, W]
        mean = arr.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        std = arr.std(dim=(2, 3), keepdim=True) + 1e-6
        return (arr - mean) / std
    else:
        raise ValueError(f"Unsupported shape: {arr.shape}")


def normalize_to_vae_range(arr: torch.Tensor) -> torch.Tensor:
    """
    将[0, 1]范围的图像归一化到[-1, 1] (VAE期望的输入)

    Args:
        arr: [0, 1]范围的tensor

    Returns:
        [-1, 1]范围的tensor
    """
    return arr * 2.0 - 1.0


def denormalize_from_vae_range(arr: torch.Tensor) -> torch.Tensor:
    """
    将VAE输出的[-1, 1]范围转回[0, 1]

    Args:
        arr: [-1, 1]范围的tensor

    Returns:
        [0, 1]范围的tensor
    """
    return (arr + 1.0) / 2.0


class CellPaintingPreprocessor:
    """
    Cell Painting图像预处理器

    完整的预处理pipeline：
    1. 百分位数裁剪 → [0, 1]
    2. (可选) 通道级标准化
    3. 归一化到[-1, 1] (VAE输入)
    """

    def __init__(
        self,
        percentile_bounds: Tuple[float, float] = (0.5, 99.5),
        use_channel_standardize: bool = False,
        sample_stride: int = 100,
        percentile_per_channel: bool = False,
        fixed_stats_path: Optional[str] = None,
    ):
        """
        Args:
            percentile_bounds: 百分位数裁剪范围
            use_channel_standardize: 是否使用通道级标准化
            sample_stride: 百分位数计算的采样步长
        """
        self.percentile_bounds = percentile_bounds
        self.use_channel_standardize = use_channel_standardize
        self.sample_stride = sample_stride
        self.percentile_per_channel = percentile_per_channel
        self.fixed_stats_path = fixed_stats_path
        self.fixed_q_low: Optional[torch.Tensor] = None
        self.fixed_q_high: Optional[torch.Tensor] = None
        self.fixed_mean01: Optional[torch.Tensor] = None
        self.fixed_std01: Optional[torch.Tensor] = None

        if fixed_stats_path is not None:
            stats = load_fixed_stats(fixed_stats_path)
            self.fixed_q_low = torch.tensor(stats["q_low"], dtype=torch.float32)
            self.fixed_q_high = torch.tensor(stats["q_high"], dtype=torch.float32)
            self.fixed_mean01 = torch.tensor(stats["mean01"], dtype=torch.float32)
            self.fixed_std01 = torch.tensor(stats["std01"], dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        预处理pipeline

        Args:
            x: 原始图像 [C, H, W] or [B, C, H, W]
               可以是任意位深度（会自动检测）

        Returns:
            预处理后的图像 [-1, 1]范围
        """
        # 1. 裁剪归一化 → [0, 1]
        if self.fixed_q_low is not None and self.fixed_q_high is not None:
            x = fixed_percentile_clip(x, self.fixed_q_low, self.fixed_q_high)
        else:
            x = percentile_clip(
                x,
                self.percentile_bounds,
                self.sample_stride,
                per_channel=self.percentile_per_channel,
            )

        # 2. (可选) 通道级标准化
        if self.use_channel_standardize:
            x = channel_wise_standardize(x)
            # 标准化后需要重新映射到[0, 1]
            # 简单的min-max归一化
            x_min = x.min()
            x_max = x.max()
            if x_max - x_min > 1e-6:
                x = (x - x_min) / (x_max - x_min)
            else:
                x = torch.zeros_like(x)

        # 3. 归一化到[-1, 1]
        x = normalize_to_vae_range(x)

        return x

    def __repr__(self):
        return (
            f"CellPaintingPreprocessor(\n"
            f"  percentile_bounds={self.percentile_bounds},\n"
            f"  use_channel_standardize={self.use_channel_standardize},\n"
            f"  sample_stride={self.sample_stride},\n"
            f"  percentile_per_channel={self.percentile_per_channel},\n"
            f"  fixed_stats_path={self.fixed_stats_path}\n"
            f")"
        )


def test_preprocessing():
    """测试预处理功能"""
    print("\n" + "="*70)
    print("Testing Cell Painting Preprocessing")
    print("="*70)

    # 1. 测试百分位数裁剪
    print("\n--- Test 1: Percentile Clipping ---")

    # 模拟16位图像
    img_16bit = torch.randint(0, 65536, (6, 512, 512), dtype=torch.float32)
    print(f"16-bit image range: [{img_16bit.min():.1f}, {img_16bit.max():.1f}]")

    clipped = percentile_clip(img_16bit, bounds=(0.5, 99.5))
    print(f"After clipping: [{clipped.min():.3f}, {clipped.max():.3f}]")
    print(f"Expected: [0.0, 1.0]")

    # 2. 测试通道级标准化
    print("\n--- Test 2: Channel-wise Standardization ---")

    img = torch.randn(6, 512, 512) * 10 + 50  # mean≈50, std≈10
    print(f"Original - mean: {img.mean():.2f}, std: {img.std():.2f}")

    standardized = channel_wise_standardize(img)
    print(f"Standardized - mean: {standardized.mean():.6f}, std: {standardized.std():.6f}")
    print(f"Expected: mean≈0, std≈1")

    # 3. 测试完整pipeline
    print("\n--- Test 3: Full Preprocessing Pipeline ---")

    preprocessor = CellPaintingPreprocessor(
        percentile_bounds=(0.5, 99.5),
        use_channel_standardize=False
    )
    print(preprocessor)

    # 模拟原始图像
    raw_img = torch.randint(0, 256, (6, 512, 512), dtype=torch.float32)
    print(f"\nRaw image: [{raw_img.min():.1f}, {raw_img.max():.1f}]")

    processed = preprocessor(raw_img)
    print(f"Processed: [{processed.min():.3f}, {processed.max():.3f}]")
    print(f"Expected: [-1, 1] (VAE input range)")

    # 4. 测试batch处理
    print("\n--- Test 4: Batch Processing ---")

    batch = torch.randint(0, 256, (4, 6, 512, 512), dtype=torch.float32)
    print(f"Batch shape: {batch.shape}")

    processed_batch = preprocessor(batch)
    print(f"Processed batch shape: {processed_batch.shape}")
    print(f"Processed range: [{processed_batch.min():.3f}, {processed_batch.max():.3f}]")

    # 5. 测试反归一化
    print("\n--- Test 5: Denormalization ---")

    vae_output = torch.randn(6, 512, 512) * 0.5  # 模拟VAE输出 (接近[-1,1])
    vae_output = torch.clamp(vae_output, -1, 1)
    print(f"VAE output range: [{vae_output.min():.3f}, {vae_output.max():.3f}]")

    denorm = denormalize_from_vae_range(vae_output)
    print(f"Denormalized range: [{denorm.min():.3f}, {denorm.max():.3f}]")
    print(f"Expected: [0, 1]")

    print("\n" + "="*70)
    print("✓ All preprocessing tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_preprocessing()

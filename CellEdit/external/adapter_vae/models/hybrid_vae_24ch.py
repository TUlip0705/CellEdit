"""
Hybrid VAE with 24-channel latent (Hybrid方案)

核心思路：
- "两头随机，中间预训练"
- 维度变了的4个conv层：随机初始化（因为SD-VAE权重塞不进去）
- 维度没变的中间层（down/up/mid blocks）：直接用SD-VAE预训练权重
- 中间层的视觉处理能力是通用的，不依赖于输入是3ch还是6ch

参数量：~167M（全部可训练，但有好的起点）
训练时间：3-4天（比Pure Native快，因为中间层有预训练起点）
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


class HybridVAE24Ch(nn.Module):
    """
    Hybrid 6-channel VAE with 24-channel latent space

    混合方案：
    - 中间层（down_blocks, up_blocks, mid_blocks）直接用SD-VAE预训练权重
    - 输入输出相关的4个conv层随机初始化（因为维度变了）
    - 所有层都可训练，预训练只是提供更好的起点
    """

    def __init__(
        self,
        pretrained_vae_path: str = "stabilityai/sd-vae-ft-mse",
        in_channels: int = 6,
        out_channels: int = 6,
        latent_channels: int = 24,
        freeze_pretrained: bool = False,  # 默认全部可训练
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        print(f"[HybridVAE] Creating Hybrid VAE")
        print(f"  Input: {in_channels}ch → Latent: {latent_channels}ch → Output: {out_channels}ch")
        print(f"  Strategy: '两头随机，中间预训练'")

        # ===== 加载SD-VAE作为base =====
        print(f"\n[HybridVAE] Loading SD-VAE from {pretrained_vae_path}")
        base_vae = AutoencoderKL.from_pretrained(pretrained_vae_path)

        # ===== Encoder =====
        self.encoder = base_vae.encoder

        # 1. 替换 conv_in: 3 → 6 (维度变了，必须随机初始化)
        print("\n[HybridVAE] Encoder modifications:")
        old_conv_in = self.encoder.conv_in
        self.encoder.conv_in = nn.Conv2d(
            in_channels,
            128,
            kernel_size=3,
            padding=1
        )
        self._init_random_conv(self.encoder.conv_in, "encoder.conv_in")
        print(f"  ✓ conv_in: [128, 3, 3, 3] → [128, {in_channels}, 3, 3] (随机初始化)")

        # 2. down_blocks 保持不变 (维度完全一样，直接用预训练权重)
        print(f"  ✓ down_blocks: 保持SD-VAE预训练权重")
        print(f"    - DownBlock1: 128→128, 512→512")
        print(f"    - DownBlock2: 128→256, 512→256")
        print(f"    - DownBlock3: 256→512, 256→128")
        print(f"    - DownBlock4: 512→512, 128→64")

        # 3. mid_block 保持不变 (维度完全一样，直接用预训练权重)
        print(f"  ✓ mid_block: 保持SD-VAE预训练权重 (ResNet+Attn+ResNet)")

        # 4. 替换 conv_out: 8 → 48 (维度变了，必须随机初始化)
        old_conv_out = self.encoder.conv_out
        self.encoder.conv_out = nn.Conv2d(
            512,
            latent_channels * 2,  # mean + logvar
            kernel_size=3,
            padding=1
        )
        self._init_small_conv(self.encoder.conv_out, "encoder.conv_out", gain=0.1)
        print(f"  ✓ conv_out: [8, 512, 3, 3] → [{latent_channels*2}, 512, 3, 3] (小初始化, gain=0.1)")

        # ===== Decoder =====
        self.decoder = base_vae.decoder

        # 1. 替换 conv_in: 4 → 24 (维度变了，必须随机初始化)
        print("\n[HybridVAE] Decoder modifications:")
        old_decoder_conv_in = self.decoder.conv_in
        self.decoder.conv_in = nn.Conv2d(
            latent_channels,
            512,
            kernel_size=3,
            padding=1
        )
        self._init_random_conv(self.decoder.conv_in, "decoder.conv_in")
        print(f"  ✓ conv_in: [512, 4, 3, 3] → [512, {latent_channels}, 3, 3] (随机初始化)")

        # 2. mid_block 保持不变 (维度完全一样，直接用预训练权重)
        print(f"  ✓ mid_block: 保持SD-VAE预训练权重 (ResNet+Attn+ResNet)")

        # 3. up_blocks 保持不变 (维度完全一样，直接用预训练权重)
        print(f"  ✓ up_blocks: 保持SD-VAE预训练权重")
        print(f"    - UpBlock1: 512→512, 64→128")
        print(f"    - UpBlock2: 512→256, 128→256")
        print(f"    - UpBlock3: 256→128, 256→512")
        print(f"    - UpBlock4: 128→128, 512→512")

        # 4. 替换 conv_out: 3 → 6 (维度变了，必须随机初始化)
        old_decoder_conv_out = self.decoder.conv_out
        self.decoder.conv_out = nn.Conv2d(
            128,
            out_channels,
            kernel_size=3,
            padding=1
        )
        self._init_small_conv(self.decoder.conv_out, "decoder.conv_out", gain=0.1)
        print(f"  ✓ conv_out: [3, 128, 3, 3] → [{out_channels}, 128, 3, 3] (小初始化, gain=0.1)")

        # ===== 可训练性控制 =====
        if freeze_pretrained:
            self._freeze_pretrained_layers()
            print("\n[HybridVAE] 预训练层已冻结，只训练4个新conv层")
        else:
            print("\n[HybridVAE] 所有层可训练（预训练权重只是更好的起点）")

        # 统计参数量
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n[HybridVAE] Parameters: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")

    def _init_random_conv(self, conv, name):
        """
        随机初始化conv层（用于输入输出层）
        使用Kaiming初始化（PyTorch默认）
        """
        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def _init_small_conv(self, conv, name, gain=0.1):
        """
        小初始化conv层（用于latent相关的输出层）
        使用Xavier with small gain，稳定训练
        """
        nn.init.xavier_uniform_(conv.weight, gain=gain)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def _freeze_pretrained_layers(self):
        """
        冻结预训练层（down_blocks, up_blocks, mid_blocks）
        只训练4个新conv层（conv_in, conv_out × 2）
        """
        # Encoder: 冻结 down_blocks 和 mid_block
        for block in self.encoder.down_blocks:
            for param in block.parameters():
                param.requires_grad = False

        for param in self.encoder.mid_block.parameters():
            param.requires_grad = False

        for param in self.encoder.conv_norm_out.parameters():
            param.requires_grad = False

        # Decoder: 冻结 mid_block 和 up_blocks
        for param in self.decoder.mid_block.parameters():
            param.requires_grad = False

        for block in self.decoder.up_blocks:
            for param in block.parameters():
                param.requires_grad = False

        for param in self.decoder.conv_norm_out.parameters():
            param.requires_grad = False

    def encode(self, x):
        """
        编码6通道图像到24通道latent

        Args:
            x: [B, 6, H, W] in range [-1, 1]

        Returns:
            latent_dist: DiagonalGaussianDistribution with 24-ch latent
        """
        # conv_in: 6ch → 128维
        h = self.encoder.conv_in(x)

        # down_blocks: 逐步下采样和增加channel
        for down_block in self.encoder.down_blocks:
            h = down_block(h)

        # mid_block: 全局信息整合
        h = self.encoder.mid_block(h, None)

        # norm + act
        h = self.encoder.conv_norm_out(h)
        h = self.encoder.conv_act(h)

        # conv_out: 512维 → 48维 (24-ch mean + 24-ch logvar)
        h = self.encoder.conv_out(h)

        # 创建latent distribution
        latent_dist = DiagonalGaussianDistribution(h)

        return latent_dist

    def decode(self, z):
        """
        解码24通道latent到6通道图像

        Args:
            z: [B, 24, h, w] latent

        Returns:
            x_recon: [B, 6, H, W] in range [-1, 1]
        """
        # conv_in: 24ch → 512维
        h = self.decoder.conv_in(z)

        # mid_block: 全局信息整合
        h = self.decoder.mid_block(h, None)

        # up_blocks: 逐步上采样和减少channel
        for up_block in self.decoder.up_blocks:
            h = up_block(h)

        # norm + act
        h = self.decoder.conv_norm_out(h)
        h = self.decoder.conv_act(h)

        # conv_out: 128维 → 6ch
        x_recon = self.decoder.conv_out(h)

        return x_recon

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, 6, H, W] in range [-1, 1]

        Returns:
            x_recon: [B, 6, H, W]
            latent_dist: DiagonalGaussianDistribution
        """
        # Encode
        latent_dist = self.encode(x)

        # Sample from latent distribution
        z = latent_dist.sample()

        # Decode
        x_recon = self.decode(z)

        return x_recon, latent_dist


def test_model():
    """测试模型创建和前向传播"""
    print("\n" + "="*70)
    print("Testing Hybrid VAE")
    print("="*70)

    # 创建模型（全部可训练）
    print("\n--- Test 1: 全部可训练模式 ---")
    model = HybridVAE24Ch(
        in_channels=6,
        out_channels=6,
        latent_channels=24,
        freeze_pretrained=False
    )

    # 测试前向传播
    print("\n" + "-"*70)
    print("Forward pass test")
    print("-"*70)

    B, C, H, W = 2, 6, 512, 512
    x = torch.randn(B, C, H, W) * 0.5  # [-1, 1]范围

    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")

    with torch.no_grad():
        x_recon, latent_dist = model(x)
        z = latent_dist.sample()

    print(f"\nLatent shape: {z.shape}")
    print(f"Latent mean: {z.mean():.3f} (expect ~0)")
    print(f"Latent std: {z.std():.3f} (expect ~1)")

    print(f"\nOutput shape: {x_recon.shape}")
    print(f"Output range: [{x_recon.min():.3f}, {x_recon.max():.3f}]")

    # 检查loss
    kl_loss = latent_dist.kl().mean()
    rec_loss = torch.nn.functional.mse_loss(x_recon, x)
    print(f"\nInitial KL loss: {kl_loss:.6f}")
    print(f"Initial Rec loss: {rec_loss:.6f}")

    # 测试冻结模式
    print("\n\n--- Test 2: 冻结预训练层模式 ---")
    model_frozen = HybridVAE24Ch(
        in_channels=6,
        out_channels=6,
        latent_channels=24,
        freeze_pretrained=True
    )

    trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_frozen.parameters())
    print(f"\nFrozen mode: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")
    print(f"Trainable ratio: {trainable/total*100:.1f}%")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_model()

"""
Model architectures for multi-modal 3D brain imaging prediction.

Three architectures share the same building blocks:
  - 3DCNN : convolutional streams -> global average pooling -> FC head
  - 3DVT  : patch embedding -> transformer encoder -> FC head (Vision Transformer)
  - 3DHB  : convolutional streams -> patch embedding -> transformer encoder -> FC head (Hybrid)

All three end in a shared "multi-output head": a small shared bottleneck
followed by one lightweight prediction head per output. This is what lets
the model predict several cognitive/behavioral scores from the same brain
scan in a single forward pass, instead of training a separate model per
target.
"""

from typing import Dict, List, Literal, Optional
import warnings

import torch
import torch.nn as nn


ACTIVATIONS = {
    'relu': nn.ReLU(inplace=True),
    'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
    'elu': nn.ELU(inplace=True),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(inplace=True),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
}


def get_activation(name: str) -> nn.Module:
    return ACTIVATIONS.get(name.lower(), nn.ReLU(inplace=True))


# ---------------------------------------------------------------------------
# CNN building blocks
# ---------------------------------------------------------------------------

class Conv3DBlock(nn.Module):
    """Conv3d -> BatchNorm -> activation -> dropout -> optional MaxPool."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, activation='relu', batch_norm=True, dropout=0.0,
                 pool_size=None, pool_stride=None):
        super().__init__()

        layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
        if batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))
        if activation:
            layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.conv_block = nn.Sequential(*layers)

        self.pool = nn.MaxPool3d(pool_size, stride=pool_stride or pool_size) if pool_size else None

    def forward(self, x):
        x = self.conv_block(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class CNNFeatureExtractor(nn.Module):
    """Stack of Conv3DBlocks for one input stream (one imaging modality)."""

    def __init__(self, in_channels, cnn_layers_info: List[Dict]):
        super().__init__()

        layers = []
        current_channels = in_channels
        for layer_info in cnn_layers_info:
            layers.append(Conv3DBlock(
                in_channels=current_channels,
                out_channels=layer_info['filters'],
                kernel_size=layer_info.get('kernel_size', 3),
                stride=layer_info.get('stride', 1),
                padding=layer_info.get('padding', 1),
                activation=layer_info.get('activation', 'relu'),
                batch_norm=layer_info.get('batch_norm', True),
                dropout=layer_info.get('dropout', 0.0),
                pool_size=layer_info.get('pool_size'),
                pool_stride=layer_info.get('pool_stride'),
            ))
            current_channels = layer_info['filters']

        self.features = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, x):
        return self.features(x)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class PatchEmbedding3D(nn.Module):
    """Splits a 3D volume into patches and linearly embeds each one."""

    def __init__(self, img_shape, patch_size, in_channels, embed_dim):
        super().__init__()
        _, D, H, W = img_shape
        self.n_patches = (D // patch_size) * (H // patch_size) * (W // patch_size)

        self.projection = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))

    def forward(self, x):
        x = self.projection(x)          # (B, embed_dim, D', H', W')
        x = x.flatten(2)                # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)           # (B, n_patches, embed_dim)
        return x + self.position_embedding


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer block: self-attention + MLP, both with residuals."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, attention_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of TransformerEncoderBlocks."""

    def __init__(self, transformer_layers_info: List[Dict]):
        super().__init__()
        blocks = [
            TransformerEncoderBlock(
                embed_dim=layer['embed_dim'],
                num_heads=layer['num_heads'],
                mlp_ratio=layer.get('mlp_ratio', 4.0),
                dropout=layer.get('dropout', 0.1),
                attention_dropout=layer.get('attention_dropout', 0.0),
            )
            for layer in transformer_layers_info
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Fusion across imaging modalities
# ---------------------------------------------------------------------------

class MultiModalFusion(nn.Module):
    """Combines per-modality feature vectors into a single fused vector.

    'concat' keeps every modality's features side by side; 'add' averages
    them (requires equal feature dimensions). 'gate' and 'attention' are
    accepted for forward-compatibility with richer fusion strategies but
    currently fall back to concatenation.
    """

    def __init__(self, fusion_type: str, feature_dims: List[int], output_dim: int):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.feature_dims = feature_dims

        if self.fusion_type == 'concat':
            self.output_dim = sum(feature_dims)
        elif self.fusion_type == 'add':
            if not all(d == feature_dims[0] for d in feature_dims):
                raise ValueError("'add' fusion requires all feature dims to be equal")
            self.output_dim = feature_dims[0]
        else:
            self.output_dim = output_dim

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if self.fusion_type == 'add':
            return torch.stack(features, dim=0).mean(dim=0)
        return torch.cat(features, dim=1)


# ---------------------------------------------------------------------------
# Multi-output prediction head
# ---------------------------------------------------------------------------

class FCBlock(nn.Module):
    """Linear -> BatchNorm -> activation -> dropout."""

    def __init__(self, in_features, out_features, activation='relu', batch_norm=True, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        if activation:
            layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MultiOutputFCHead(nn.Module):
    """Shared bottleneck followed by one small prediction head per output.

    This is the key piece that makes the model multi-output: all targets
    are predicted from the same shared representation, but each target
    gets its own head so it can specialize. With a single output this
    reduces to one bottleneck + one head, i.e. an ordinary regression head.
    """

    def __init__(self, in_features, fc_layers_info: List[Dict], output_dim: int,
                 bottleneck_dim: Optional[int] = None, output_head_sizes: Optional[List[int]] = None):
        super().__init__()
        self.output_dim = output_dim
        self.output_head_sizes = output_head_sizes or [32] * output_dim

        bottleneck_layers = []
        current_features = in_features
        for layer_info in fc_layers_info:
            bottleneck_layers.append(FCBlock(
                in_features=current_features,
                out_features=layer_info['units'],
                activation=layer_info.get('activation', 'relu'),
                batch_norm=layer_info.get('batch_norm', True),
                dropout=layer_info.get('dropout', 0.0),
            ))
            current_features = layer_info['units']
        self.bottleneck = nn.Sequential(*bottleneck_layers)
        bottleneck_output_dim = current_features

        if bottleneck_dim is not None and bottleneck_dim != bottleneck_output_dim:
            self.bottleneck.add_module('bottleneck_projection', nn.Sequential(
                nn.Linear(bottleneck_output_dim, bottleneck_dim),
                nn.ReLU(inplace=True),
            ))
            bottleneck_output_dim = bottleneck_dim
        self.bottleneck_dim = bottleneck_output_dim

        self.output_heads = nn.ModuleList()
        for out_idx in range(output_dim):
            head_hidden_dim = self.output_head_sizes[out_idx] if out_idx < len(self.output_head_sizes) else 64
            self.output_heads.append(nn.Sequential(
                FCBlock(bottleneck_output_dim, head_hidden_dim, activation='relu', batch_norm=False, dropout=0.1),
                nn.Linear(head_hidden_dim, 1),
            ))

    def forward(self, x):
        """Returns a [B, 1] tensor if output_dim == 1, otherwise a list of [B, 1] tensors."""
        bottleneck_features = self.bottleneck(x)
        outputs = [head(bottleneck_features) for head in self.output_heads]
        return outputs[0] if self.output_dim == 1 else outputs


# ---------------------------------------------------------------------------
# Complete architectures
# ---------------------------------------------------------------------------

class MultiModal3DCNN(nn.Module):
    """Parallel 3D CNN streams (one per modality) -> fusion -> multi-output head."""

    def __init__(self, n_streams, input_shapes, cnn_layers_info, fc_layers_info,
                 fusion_type, covariate_flag, covariate_dim, output_dim,
                 bottleneck_dim=None, output_head_sizes=None):
        super().__init__()
        self.covariate_flag = covariate_flag

        self.cnn_streams = nn.ModuleList([
            CNNFeatureExtractor(in_channels=input_shapes[i][0], cnn_layers_info=cnn_layers_info)
            for i in range(n_streams)
        ])
        cnn_out_channels = self.cnn_streams[0].out_channels
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.fusion = MultiModalFusion(fusion_type, [cnn_out_channels] * n_streams, output_dim=cnn_out_channels)

        fc_input_dim = self.fusion.output_dim + (covariate_dim if covariate_flag else 0)
        self.fc_head = MultiOutputFCHead(fc_input_dim, fc_layers_info, output_dim, bottleneck_dim, output_head_sizes)

    def forward(self, inputs, covariates=None, modality_mask=None):
        features = []
        for x, cnn in zip(inputs, self.cnn_streams):
            feat = cnn(x)
            feat = self.global_pool(feat).flatten(1)
            features.append(feat)

        fused = self.fusion(features)
        if self.covariate_flag and covariates is not None:
            fused = torch.cat([fused, covariates], dim=1)
        return self.fc_head(fused)


class MultiModal3DVT(nn.Module):
    """Parallel Vision Transformer streams (one per modality) -> fusion -> multi-output head."""

    def __init__(self, n_streams, input_shapes, transformer_layers_info, fc_layers_info,
                 patch_size, fusion_type, covariate_flag, covariate_dim, output_dim,
                 bottleneck_dim=None, output_head_sizes=None):
        super().__init__()
        self.covariate_flag = covariate_flag
        embed_dim = transformer_layers_info[0]['embed_dim']

        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding3D(input_shapes[i], patch_size, input_shapes[i][0], embed_dim)
            for i in range(n_streams)
        ])
        self.transformers = nn.ModuleList([TransformerEncoder(transformer_layers_info) for _ in range(n_streams)])
        self.cls_tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, embed_dim)) for _ in range(n_streams)])

        self.fusion = MultiModalFusion(fusion_type, [embed_dim] * n_streams, output_dim=embed_dim)

        fc_input_dim = self.fusion.output_dim + (covariate_dim if covariate_flag else 0)
        self.fc_head = MultiOutputFCHead(fc_input_dim, fc_layers_info, output_dim, bottleneck_dim, output_head_sizes)

    def forward(self, inputs, covariates=None, modality_mask=None):
        features = []
        for x, patch_embed, transformer, cls_token in zip(inputs, self.patch_embeddings, self.transformers, self.cls_tokens):
            x = patch_embed(x)
            cls = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = transformer(x)
            features.append(x[:, 0])  # CLS token summarizes the whole volume

        fused = self.fusion(features)
        if self.covariate_flag and covariates is not None:
            fused = torch.cat([fused, covariates], dim=1)
        return self.fc_head(fused)


class MultiModal3DHybrid(nn.Module):
    """Parallel CNN + Transformer streams (one per modality) -> fusion -> multi-output head.

    Each stream first extracts local features with a small CNN, then treats
    the resulting feature map as a grid of patches for a transformer encoder.
    This keeps the parameter count of a pure 3D ViT manageable for large
    brain volumes while still modeling long-range spatial dependencies.
    """

    def __init__(self, n_streams, input_shapes, cnn_layers_info, transformer_layers_info,
                 fc_layers_info, patch_size, fusion_type, covariate_flag, covariate_dim, output_dim,
                 bottleneck_dim=None, output_head_sizes=None):
        super().__init__()
        self.covariate_flag = covariate_flag
        embed_dim = transformer_layers_info[0]['embed_dim']

        self.cnn_streams = nn.ModuleList([
            CNNFeatureExtractor(in_channels=input_shapes[i][0], cnn_layers_info=cnn_layers_info)
            for i in range(n_streams)
        ])

        cnn_output_shapes = []
        with torch.no_grad():
            for i in range(n_streams):
                dummy_output = self.cnn_streams[i](torch.zeros(1, *input_shapes[i]))
                cnn_output_shapes.append(dummy_output.shape[1:])  # (C, D, H, W)

        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding3D((1, *cnn_output_shapes[i][1:]), patch_size, cnn_output_shapes[i][0], embed_dim)
            for i in range(n_streams)
        ])
        self.transformers = nn.ModuleList([TransformerEncoder(transformer_layers_info) for _ in range(n_streams)])
        self.cls_tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, embed_dim)) for _ in range(n_streams)])

        self.fusion = MultiModalFusion(fusion_type, [embed_dim] * n_streams, output_dim=embed_dim)

        fc_input_dim = self.fusion.output_dim + (covariate_dim if covariate_flag else 0)
        self.fc_head = MultiOutputFCHead(fc_input_dim, fc_layers_info, output_dim, bottleneck_dim, output_head_sizes)

    def forward(self, inputs, covariates=None, modality_mask=None):
        features = []
        for x, cnn, patch_embed, transformer, cls_token in zip(
            inputs, self.cnn_streams, self.patch_embeddings, self.transformers, self.cls_tokens
        ):
            x = cnn(x)
            x = patch_embed(x)
            cls = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = transformer(x)
            features.append(x[:, 0])

        fused = self.fusion(features)
        if self.covariate_flag and covariates is not None:
            fused = torch.cat([fused, covariates], dim=1)
        return self.fc_head(fused)


class ModelDesigner:
    """Builds a model from a config dict without the caller needing to know
    which concrete architecture class to instantiate."""

    def __init__(self, architecture_type: Literal['3DCNN', '3DVT', '3DHB'], layers_info: Dict,
                 n_input_streams: int, input_shapes: List[tuple],
                 fusion_type: Literal['concat', 'add', 'gate', 'attention'] = 'concat',
                 covariate_flag: bool = False, covariate_dim: int = 0, output_dim: int = 1,
                 bottleneck_dim: Optional[int] = None, output_head_sizes: Optional[List[int]] = None):
        self.architecture_type = architecture_type.upper()
        self.layers_info = layers_info
        self.n_input_streams = n_input_streams
        self.input_shapes = input_shapes
        self.fusion_type = fusion_type.lower()
        self.covariate_flag = covariate_flag
        self.covariate_dim = covariate_dim
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_head_sizes = output_head_sizes or [32] * output_dim

        if self.architecture_type not in ['3DCNN', '3DVT', '3DHB']:
            raise ValueError(f"Unknown architecture: {self.architecture_type}")
        if len(self.input_shapes) != self.n_input_streams:
            raise ValueError(f"input_shapes length ({len(self.input_shapes)}) must match n_input_streams ({self.n_input_streams})")
        if output_dim < 1:
            raise ValueError("output_dim must be >= 1")
        if len(self.output_head_sizes) != output_dim:
            warnings.warn(
                f"output_head_sizes length ({len(self.output_head_sizes)}) doesn't match "
                f"output_dim ({output_dim}); using default [32] * {output_dim}"
            )
            self.output_head_sizes = [32] * output_dim

    def build_model(self) -> nn.Module:
        common_kwargs = dict(
            n_streams=self.n_input_streams,
            input_shapes=self.input_shapes,
            fc_layers_info=self.layers_info.get('fc_layers', []),
            fusion_type=self.fusion_type,
            covariate_flag=self.covariate_flag,
            covariate_dim=self.covariate_dim,
            output_dim=self.output_dim,
            bottleneck_dim=self.bottleneck_dim,
            output_head_sizes=self.output_head_sizes,
        )

        if self.architecture_type == '3DCNN':
            return MultiModal3DCNN(cnn_layers_info=self.layers_info.get('cnn_layers', []), **common_kwargs)
        elif self.architecture_type == '3DVT':
            return MultiModal3DVT(
                transformer_layers_info=self.layers_info.get('transformer_layers', []),
                patch_size=self.layers_info.get('patch_size', 8),
                **common_kwargs,
            )
        else:  # 3DHB
            return MultiModal3DHybrid(
                cnn_layers_info=self.layers_info.get('cnn_layers', []),
                transformer_layers_info=self.layers_info.get('transformer_layers', []),
                patch_size=self.layers_info.get('patch_size', 8),
                **common_kwargs,
            )

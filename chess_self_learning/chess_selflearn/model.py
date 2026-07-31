from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import Tensor, nn


@dataclass(slots=True)
class ModelConfig:
    input_planes: int = 34
    channels: int = 128
    residual_blocks: int = 8
    policy_planes: int = 73
    value_hidden: int = 256
    normalization: str = "batchnorm"
    activation: str = "relu"


def make_activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    if name.lower() == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def make_norm(name: str, channels: int) -> nn.Module:
    if name.lower() == "batchnorm":
        return nn.BatchNorm2d(channels)
    if name.lower() == "groupnorm":
        groups = min(32, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported normalization: {name}")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, normalization: str, activation: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = make_norm(normalization, channels)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = make_norm(normalization, channels)
        self.act2 = make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act2(x + residual)


class ChessPolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_planes, config.channels, 3, padding=1, bias=False),
            make_norm(config.normalization, config.channels),
            make_activation(config.activation),
        )
        self.tower = nn.Sequential(
            *[
                ResidualBlock(
                    config.channels,
                    config.normalization,
                    config.activation,
                )
                for _ in range(config.residual_blocks)
            ]
        )
        self.policy_head = nn.Conv2d(config.channels, config.policy_planes, 1)
        self.value_conv = nn.Sequential(
            nn.Conv2d(config.channels, 32, 1, bias=False),
            make_norm(config.normalization, 32),
            make_activation(config.activation),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, config.value_hidden),
            make_activation(config.activation),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        final_value = self.value_head[-2]
        assert isinstance(final_value, nn.Linear)
        nn.init.uniform_(final_value.weight, -1e-3, 1e-3)
        nn.init.zeros_(final_value.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.tower(self.stem(x))
        policy = self.policy_head(x).permute(0, 2, 3, 1).flatten(start_dim=1)
        value = self.value_head(self.value_conv(x)).squeeze(1)
        return policy, value

    def export_config(self) -> dict[str, object]:
        return asdict(self.config)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B A x, with W and b frozen.

    Shapes: W (out, in), A (r, in), B (out, r).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        init_std: float | None = None,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

        in_f, out_f = base.in_features, base.out_features
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=dt))
        # Paper: Gaussian A, zero B. The variance is unspecified; the default matches
        # the variance of kaiming-uniform(a=sqrt(5)) and does not depend on r.
        self.init_std = init_std if init_std is not None else 1.0 / math.sqrt(3 * in_f)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.lora_A, mean=0.0, std=self.init_std)
        nn.init.zeros_(self.lora_B)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return out + update * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_weight().to(self.base.weight.dtype)
            self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, merged={self.merged}"

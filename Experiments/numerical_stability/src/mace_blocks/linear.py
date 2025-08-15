import torch
from e3nn import o3
import cuequivariance as cue
import cuequivariance_torch as cuet

class LinearLayer(torch.nn.Module):
    """
    Thin wrapper over cuet.Linear or e3nn.o3.Linear with a unified constructor
    and a single-forward signature: y = W x.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        cueq_config=None,          # your CuEquivarianceConfig
        shared_weights: bool = True,
        internal_weights: bool = True,
        use_cueq: bool = True,
        math_dtype: torch.dtype,
        device: torch.device
    ):
        super().__init__()
        if use_cueq:
            self.impl = cuet.Linear(
                cue.Irreps("O3", irreps_in),
                cue.Irreps("O3", irreps_out),
                layout="mul_ir",
                shared_weights=shared_weights,
                math_dtype=math_dtype,
                use_fallback=False,
            ).to(device)
        else:
            self.impl = o3.Linear(
                irreps_in,
                irreps_out,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
            ).to(device)
            # Ensure parameters are in the correct dtype
            for param in self.impl.parameters():
                param.data = param.data.to(math_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor of shape [..., irreps_in.dim]
        returns y : Tensor of shape [..., irreps_out.dim]
        """
        output = self.impl(x)
        # Ensure output is in the same dtype as input
        if output.dtype != x.dtype:
            output = output.to(x.dtype)
        return output

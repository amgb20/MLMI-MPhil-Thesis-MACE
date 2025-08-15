import torch
from e3nn import o3
import cuequivariance as cue
import cuequivariance_torch as cuet 

import mace.modules.symmetric_contraction as SymmetricContraction

class SymmetricContractionWrapper(torch.nn.Module):
    """
    Thin wrapper over cuet.SymmetricContraction and SymmetricContraction 
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        cueq_config=None,          # your CuEquivarianceConfig
        shared_weights: bool = True,
        internal_weights: bool = True,
        correlation: int,
        num_elements: int,
        use_cueq: bool = True,
        math_dtype: torch.dtype,
        device: torch.device
    ):
        super().__init__()
        self.is_cueq = bool(use_cueq)
        if self.is_cueq:
            self.sc = cuet.SymmetricContraction(
                cue.Irreps('O3', irreps_in),
                cue.Irreps('O3', irreps_out),
                layout_in=cue.ir_mul,
                layout_out=cue.mul_ir,
                contraction_degree=correlation,
                num_elements=num_elements,
                original_mace=True,
                dtype=math_dtype,
                math_dtype=math_dtype,
               ).to(device)
        else:
            prev_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(math_dtype)
                self.sc = SymmetricContraction.SymmetricContraction(
                    irreps_in=irreps_in,
                    irreps_out=irreps_out,
                    correlation=correlation,
                    num_elements=num_elements,
                ).to(device)
            finally:
                torch.set_default_dtype(prev_dtype)
            # Ensure parameters are in the correct dtype
            for param in self.sc.parameters():
                param.data = param.data.to(math_dtype)

    def forward(self, x: torch.Tensor, attrs_one_hot: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor with shape [N, C, D] (mul_ir layout); attrs_one_hot: [N, num_elements]
        returns y : Tensor of shape [N, irreps_out.dim]
        """
        # Cast inputs to module dtype to avoid dtype mismatches
        param_example = next(self.sc.parameters(), None)
        target_dtype = param_example.dtype if param_example is not None else x.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if attrs_one_hot.dtype != target_dtype:
            attrs_one_hot = attrs_one_hot.to(target_dtype)
        if self.is_cueq:
            # Match blocks.py: convert mul_ir [N, C, D] -> ir_mul [N, D, C], then flatten
            x = x.transpose(1, 2).contiguous()
            x_in = x.flatten(1).contiguous()
            # Convert one-hot attrs to integer indices
            indices = torch.nonzero(attrs_one_hot)[:, 1].to(torch.int32)
            output = self.sc(x_in, indices=indices)
        else:
            # e3nn expects (features [N, C, D], one-hot attrs [N, E])
            output = self.sc(x, attrs_one_hot)
        # Ensure output is in the same dtype as input
        if output.dtype != x.dtype:
            output = output.to(x.dtype)
        return output

import types
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet

def with_cueq_conv_fusion(conv_tp: torch.nn.Module) -> torch.nn.Module:
    """Wraps a cuet.ConvTensorProduct to use conv fusion"""
    conv_tp.original_forward = conv_tp.forward

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        return self.original_forward(
            [tp_weights, node_feats, edge_attrs],
            {1: sender},
            {0: node_feats},
            {0: receiver},
        )

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp

    # build cue descriptors + polynomial modules which allows us to use the cuequivariance library and directly conv_tp without
# having to write our own custom conv_tp config file which is annoying atm
def make_poly(in_ir, attr_ir, out_ir, math_dtype, device):
    desc = cue.descriptors.channelwise_tensor_product(
        cue.Irreps("O3", in_ir),
        cue.Irreps("O3", attr_ir),
        cue.Irreps("O3", out_ir),
    )
    return cuet.SegmentedPolynomial(
        desc.flatten_coefficient_modes()
            .squeeze_modes()
            .polynomial,
        math_dtype=math_dtype,
        output_dtype_map=[-1]
    ).to(device)

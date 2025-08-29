import torch

_DTYPE_MAP = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    torch.float64: torch.float64,
    torch.float32: torch.float32,
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
}

def _get_parent_and_attr(root, dotted):
    p = root
    parts = dotted.split(".")
    for t in parts[:-1]:
        p = getattr(p, t)
    return p, parts[-1]

def _resolve_target_dtype(module, default_target):
    """If the module advertises a layer_dtype, use it; else use default_target."""
    ld = getattr(module, "layer_dtype", None)
    if ld is None:
        return default_target
    return _DTYPE_MAP.get(ld, default_target)

def normalize_fp64_only(model: torch.nn.Module, default_target=torch.float32):
    """
    Convert ONLY float64 params/buffers to a target dtype.
    If a module has .layer_dtype (e.g., 'float16'), that wins for tensors in that module.
    Otherwise, use default_target (e.g., torch.float32).
    """
    # 1) Parameters
    for name, p in model.named_parameters(recurse=True):
        if not p.is_floating_point() or p.dtype != torch.float64:
            continue
        parent, attr = _get_parent_and_attr(model, name)
        tgt = _resolve_target_dtype(parent, default_target)
        if p.data.dtype != tgt:
            p.data = p.data.to(tgt)
            # (params don't need re-registration)

    # 2) Buffers
    # Try direct register_buffer; if that fails (ScriptModule), fall back to state_dict reload.
    to_update_in_sd = {}
    for name, b in model.named_buffers(recurse=True):
        if not b.is_floating_point() or b.dtype != torch.float64:
            continue
        parent, attr = _get_parent_and_attr(model, name)
        tgt = _resolve_target_dtype(parent, default_target)
        new_b = b.to(tgt)
        try:
            parent.register_buffer(attr, new_b, persistent=True)
        except Exception:
            to_update_in_sd[name] = new_b

    if to_update_in_sd:
        sd = model.state_dict()
        for k, v in to_update_in_sd.items():
            sd[k] = v
        model.load_state_dict(sd)

def audit_model_dtypes(model):
    from collections import Counter
    pc = Counter(p.dtype for p in model.parameters())
    bc = Counter(b.dtype for b in model.buffers())
    print("Param dtypes:", pc)
    print("Buffer dtypes:", bc)

from __future__ import annotations

import os
from collections.abc import Callable

import torch
from torch import Tensor


_NS5_COMPILED_CACHE: dict[tuple[int, float], Callable[[Tensor], Tensor] | None] = {}
_POLAR_EXPRESS_COMPILED_CACHE: dict[tuple[int, float], Callable[[Tensor], Tensor] | None] = {}

POLAR_EXPRESS_COEFFS: tuple[tuple[float, float, float], ...] = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)


def _compile_target(
    target: Callable[[Tensor], Tensor],
    *,
    dynamic: bool | None = None,
    fullgraph: bool | None = None,
) -> Callable[[Tensor], Tensor]:
    if not hasattr(torch, "compile"):
        return target
    if not bool(int(os.environ.get("COMPILE_ENABLED", "1"))):
        return target

    kwargs = {
        "dynamic": bool(int(os.environ.get("COMPILE_DYNAMIC", "0"))) if dynamic is None else dynamic,
        "fullgraph": bool(int(os.environ.get("COMPILE_FULLGRAPH", "1"))) if fullgraph is None else fullgraph,
    }
    backend = os.environ.get("COMPILE_BACKEND", "").strip()
    mode = os.environ.get("COMPILE_MODE", "").strip()
    if backend:
        kwargs["backend"] = backend
    if mode:
        kwargs["mode"] = mode
    return torch.compile(target, **kwargs)


def _zeropower_via_newtonschulz5_impl(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Batched Newton-Schulz orthogonalization. Accepts (B, M, N) or (M, N)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)

    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Run NS5, using a compiled CUDA path when available."""
    use_compiled = G.is_cuda and bool(int(os.environ.get("MUON_COMPILE_NS5", "1")))
    if not use_compiled:
        return _zeropower_via_newtonschulz5_impl(G, steps=steps, eps=eps)

    cache_key = (steps, eps)
    compiled_fn = _NS5_COMPILED_CACHE.get(cache_key)
    if cache_key not in _NS5_COMPILED_CACHE:

        def _compiled_ns5(x: Tensor) -> Tensor:
            return _zeropower_via_newtonschulz5_impl(x, steps=steps, eps=eps)

        try:
            compiled_fn = _compile_target(_compiled_ns5, dynamic=False, fullgraph=True)
        except Exception:
            compiled_fn = None
        _NS5_COMPILED_CACHE[cache_key] = compiled_fn

    if compiled_fn is None:
        return _zeropower_via_newtonschulz5_impl(G, steps=steps, eps=eps)

    try:
        return compiled_fn(G)
    except Exception:
        _NS5_COMPILED_CACHE[cache_key] = None
        return _zeropower_via_newtonschulz5_impl(G, steps=steps, eps=eps)


def _zeropower_via_polarexpress5_impl(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Degree-5 Polar Express orthogonalization with the safety scaling used in benchmarks."""
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)

    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + eps)
    coeffs = POLAR_EXPRESS_COEFFS[:steps]
    if steps > len(POLAR_EXPRESS_COEFFS):
        coeffs = coeffs + (POLAR_EXPRESS_COEFFS[-1],) * (steps - len(POLAR_EXPRESS_COEFFS))

    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


def zeropower_via_polarexpress5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Run Polar Express, using a compiled CUDA path when available."""
    use_compiled = G.is_cuda and bool(
        int(os.environ.get("MUON_COMPILE_POLAR", os.environ.get("MUON_COMPILE_NS5", "1")))
    )
    if not use_compiled:
        return _zeropower_via_polarexpress5_impl(G, steps=steps, eps=eps)

    cache_key = (steps, eps)
    compiled_fn = _POLAR_EXPRESS_COMPILED_CACHE.get(cache_key)
    if cache_key not in _POLAR_EXPRESS_COMPILED_CACHE:

        def _compiled_polar(x: Tensor) -> Tensor:
            return _zeropower_via_polarexpress5_impl(x, steps=steps, eps=eps)

        try:
            compiled_fn = _compile_target(_compiled_polar, dynamic=False, fullgraph=True)
        except Exception:
            compiled_fn = None
        _POLAR_EXPRESS_COMPILED_CACHE[cache_key] = compiled_fn

    if compiled_fn is None:
        return _zeropower_via_polarexpress5_impl(G, steps=steps, eps=eps)

    try:
        return compiled_fn(G)
    except Exception:
        _POLAR_EXPRESS_COMPILED_CACHE[cache_key] = None
        return _zeropower_via_polarexpress5_impl(G, steps=steps, eps=eps)


def zeropower_via_muon_backend(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    impl = os.environ.get("MUON_ORTHO_IMPL", "ns5").strip().lower()
    if impl in ("ns5", "newton-schulz", "newtonschulz"):
        return zeropower_via_newtonschulz5(G, steps=steps, eps=eps)
    if impl in ("polar", "polar-express", "polar_express"):
        return zeropower_via_polarexpress5(G, steps=steps, eps=eps)
    raise ValueError(f"Unknown MUON_ORTHO_IMPL={impl!r}")


__all__ = [
    "POLAR_EXPRESS_COEFFS",
    "zeropower_via_muon_backend",
    "zeropower_via_newtonschulz5",
    "zeropower_via_polarexpress5",
]

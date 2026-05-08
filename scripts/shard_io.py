"""
Shard I/O helpers — read/write embedding shards.

Default write format (when `zstandard` is installed): zstd-compressed safetensors.
Fallback without zstandard: plain safetensors. Fallback without safetensors: .pt.

Layout for one shard "all_steps_shard_N":
    all_steps_shard_N.safetensors.zst   — zstd-compressed tensors (default)
    all_steps_shard_N.ids.json.zst      — zstd-compressed doc-id list (default)

    Uncompressed (when zstandard missing):
    all_steps_shard_N.safetensors       — tensors (repr_hidden, sparse_indices, sparse_values)
    all_steps_shard_N.ids.json          — list of doc ID strings

    Legacy (torch.save pickle; read-only fallback):
    all_steps_shard_N.pt

load_shard / list_shards / shard_exists probe all three layouts transparently;
no caller ever needs to know the extension. list_shards prefers uncompressed
over compressed when both are present (avoids the decompress step).
"""
import json
from pathlib import Path

import torch


def pool_sparse_across_k(si: torch.Tensor, sv: torch.Tensor,
                          vocab_size: int = None, batch_size: int = 256,
                          device: str = 'cpu'):
    """Max-pool sparse representations across K positions (lossless vs current eval).

    si: [N, K, topk] int indices (vocab IDs)
    sv: [N, K, topk] float/bf16 values (≥ 0)

    Vectorized via scatter_reduce + topk — no Python row loop.

    Returns:
        pooled_si: [N, 1, max_count] int32  (K=1 fake dim for legacy compatibility)
        pooled_sv: [N, 1, max_count] bfloat16
    """
    N, K, topk = si.shape
    if K == 1:
        return si.to(torch.int32), sv.to(torch.bfloat16)

    if vocab_size is None:
        vocab_size = int(si.max().item()) + 1

    # Upper bound on unique terms per doc: K*topk (before dedup)
    out_top = K * topk

    si_dev = si.to(device=device, dtype=torch.long).reshape(N, K * topk)
    sv_dev = sv.to(device=device, dtype=torch.float32).reshape(N, K * topk)

    all_idxs = []
    all_vals = []
    global_max = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        pooled = torch.zeros(B, vocab_size, dtype=torch.float32, device=device)
        idx_b = si_dev[start:end].clamp(0, vocab_size - 1)
        val_b = sv_dev[start:end]
        pooled.scatter_reduce_(1, idx_b, val_b, reduce='amax', include_self=True)

        # Vectorized top-K per row: pick top `out_top` values (upper bound on non-zeros)
        k_val = min(out_top, vocab_size)
        top_vals, top_idxs = pooled.topk(k_val, dim=-1)  # [B, k_val]

        # Trim trailing zero columns (after topk, zeros are at the end since sorted desc)
        # Find the last column with any non-zero value across the batch
        any_nz = (top_vals > 0).any(dim=0)
        batch_max = int(any_nz.nonzero().max().item()) + 1 if any_nz.any() else 0
        top_vals = top_vals[:, :batch_max].contiguous()
        top_idxs = top_idxs[:, :batch_max].to(torch.int32).contiguous()

        # Mask out zero-value positions with index -1 (eval code skips these via `raw >= 0`)
        top_idxs.masked_fill_(top_vals == 0, -1)

        all_idxs.append(top_idxs.cpu() if device != 'cpu' else top_idxs)
        all_vals.append(top_vals.cpu() if device != 'cpu' else top_vals)
        global_max = max(global_max, batch_max)
        del pooled

    # Pad all batches to global_max
    final_idxs = torch.full((N, global_max), -1, dtype=torch.int32)
    final_vals = torch.zeros(N, global_max, dtype=torch.float32)
    cursor = 0
    for bi, bv in zip(all_idxs, all_vals):
        b = bi.shape[0]
        w = bi.shape[1]
        final_idxs[cursor:cursor + b, :w] = bi
        final_vals[cursor:cursor + b, :w] = bv
        cursor += b

    return final_idxs.unsqueeze(1), final_vals.to(torch.bfloat16).unsqueeze(1)

try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load
    from safetensors.torch import load as _st_load_bytes   # decode from raw bytes
    from safetensors.torch import save as _st_save_bytes   # encode to raw bytes (for zstd)
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import zstandard as _zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


def _read_maybe_zst(path_st: Path, path_zst: Path) -> bytes:
    """Read a file that may be plain or zstd-compressed. Returns raw bytes.

    Uses zstd streaming decompression — works whether the frame header
    carries the uncompressed content size (typical for whole-buffer
    `ZstdCompressor.compress(...)` output) or not (stream-written files
    from `stream_writer` without an explicit `size=` hint).
    """
    if path_st.exists():
        with open(path_st, 'rb') as f:
            return f.read()
    if path_zst.exists():
        if not HAS_ZSTD:
            raise RuntimeError(
                f"{path_zst} is zstd-compressed but 'zstandard' is not installed. "
                "Run: pip install zstandard"
            )
        with open(path_zst, 'rb') as f:
            dctx = _zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                return reader.read()
    raise FileNotFoundError(f"neither {path_st} nor {path_zst} exists")


def _atomic_write(path: str, data: bytes) -> None:
    """Write bytes atomically via temp file + rename (crash-safe)."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
    import os as _os
    _os.replace(tmp, path)


def save_shard(path, payload: dict, use_safetensors: bool = True,
               compress: bool = True, compress_level: int = 3,
               zstd_threads: int = 0) -> str:
    """Save a shard payload. Returns the actual path written (string).

    path: str or Path (with .pt or .safetensors extension, or no extension).
    payload: dict with 'ids' (list[str]) and tensor keys.
    compress: if True and `zstandard` is installed, write .safetensors.zst +
        .ids.json.zst directly. Falls back silently to uncompressed .safetensors
        when zstandard missing.
    compress_level: 1 (fast) .. 22 (max). Default 3 — on bf16 tensors, level 3
        gets within ~5% of max ratio at several× the speed of level 9.
    zstd_threads: 0 → single-threaded (default); -1 → all cores; N → N threads.
        Multi-threaded zstd compression gives 4-8× speedup on large tensors
        (>1 GB) at essentially no ratio cost.  Safe on any build of the
        `zstandard` binding — newer versions enable it by default, older
        ones silently fall back to single-threaded.
    """
    path = Path(path)
    # Strip extension to determine base name
    if path.suffix in ('.pt', '.safetensors'):
        base = path.with_suffix('')
    else:
        base = path

    if use_safetensors and HAS_SAFETENSORS:
        tensors = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
        # Make contiguous — safetensors requires it
        tensors = {k: v.contiguous() for k, v in tensors.items()}

        if compress and HAS_ZSTD:
            # Serialize to bytes, compress in-memory (multi-threaded zstd when
            # requested), atomic-write .zst.
            try:
                cctx = _zstd.ZstdCompressor(level=compress_level,
                                             threads=zstd_threads)
            except TypeError:
                # Very old `zstandard` build without `threads` kwarg — fall back.
                cctx = _zstd.ZstdCompressor(level=compress_level)
            raw = _st_save_bytes(tensors)
            out_path = str(base) + '.safetensors.zst'
            _atomic_write(out_path, cctx.compress(raw))
            if 'ids' in payload:
                ids_bytes = json.dumps(payload['ids']).encode('utf-8')
                _atomic_write(str(base) + '.ids.json.zst', cctx.compress(ids_bytes))
            return out_path
        else:
            out_path = str(base) + '.safetensors'
            _st_save(tensors, out_path)
            if 'ids' in payload:
                with open(str(base) + '.ids.json', 'w') as f:
                    json.dump(payload['ids'], f)
            return out_path
    else:
        out_path = str(base) + '.pt'
        torch.save(payload, out_path)
        return out_path


def load_shard(path, map_location='cpu') -> dict:
    """Load a shard. Auto-detects format: .safetensors, .safetensors.zst, or .pt.

    path: str or Path to shard file (either .pt or .safetensors; extension optional;
          .zst suffix is handled transparently).
    """
    path = Path(path)
    # Strip all known extensions to get the base
    name = path.name
    for suffix in ('.safetensors.zst', '.safetensors', '.pt', '.zst'):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    base = path.with_name(name)

    st_path = Path(str(base) + '.safetensors')
    zst_path = Path(str(base) + '.safetensors.zst')
    pt_path = Path(str(base) + '.pt')

    # safetensors (optionally zstd-compressed)
    if (st_path.exists() or zst_path.exists()) and HAS_SAFETENSORS:
        raw = _read_maybe_zst(st_path, zst_path)
        tensors = _st_load_bytes(raw)
        # Move to requested device (load() returns CPU tensors by default)
        if map_location != 'cpu':
            tensors = {k: v.to(map_location) for k, v in tensors.items()}
        payload = dict(tensors)

        # ids.json — also try zstd-compressed variant
        ids_path = Path(str(base) + '.ids.json')
        ids_zst = Path(str(base) + '.ids.json.zst')
        if ids_path.exists():
            with open(ids_path) as f:
                payload['ids'] = json.load(f)
        elif ids_zst.exists():
            if not HAS_ZSTD:
                raise RuntimeError(
                    f"{ids_zst} is zstd-compressed but 'zstandard' is not installed."
                )
            with open(ids_zst, 'rb') as f:
                dctx = _zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    raw_ids = reader.read()
            payload['ids'] = json.loads(raw_ids)
        return payload

    if pt_path.exists():
        return torch.load(str(pt_path), map_location=map_location)

    raise FileNotFoundError(
        f"No shard found at {base}.{{pt,safetensors,safetensors.zst}}"
    )


def list_shards(corpus_dir, prefix='all_steps_shard_'):
    """List all shards in a dir (.safetensors, .safetensors.zst, or .pt).

    Prefers uncompressed .safetensors when both .safetensors and .safetensors.zst
    exist for the same shard (no extra decompression cost).

    Returns list of base paths (without extension).
    """
    corpus_dir = Path(corpus_dir)
    # Priority (higher wins): safetensors > safetensors.zst > pt
    priority = {'.safetensors': 3, '.safetensors.zst': 2, '.pt': 1}
    seen: dict = {}   # shard_idx -> (priority, base_path)
    for ext, pri in priority.items():
        for p in corpus_dir.glob(f'{prefix}*{ext}'):
            # Strip ext (handle double-suffix .safetensors.zst)
            name = p.name[: -len(ext)]
            try:
                idx = int(name.split('_')[-1])
            except ValueError:
                continue
            if idx not in seen or seen[idx][0] < pri:
                seen[idx] = (pri, p.with_name(name))
    return [seen[i][1] for i in sorted(seen.keys())]


def shard_exists(corpus_dir, shard_id: int, prefix='all_steps_shard_') -> bool:
    """Check if a shard exists in any supported format."""
    corpus_dir = Path(corpus_dir)
    for ext in ('.safetensors', '.safetensors.zst', '.pt'):
        if (corpus_dir / f'{prefix}{shard_id}{ext}').exists():
            return True
    return False

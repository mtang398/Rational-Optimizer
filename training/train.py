import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import time
from array import array
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import transformers


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def precompute_rope(seq_len, head_dim, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]


def apply_rope(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, seq_len):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        cos, sin = precompute_rope(seq_len, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        batch, seq_len, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        cos = self.rope_cos[:, :, :seq_len, :].to(dtype=q.dtype)
        sin = self.rope_sin[:, :, :seq_len, :].to(dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(y)

GRAIN_ACTIVATION = "grain"
_LEGACY_GRAIN_ACTIVATION = "rlb_fused_global_rational"
ACTIVATIONS = ("silu", GRAIN_ACTIVATION)
_ACCEPTED_ACTIVATIONS = (*ACTIVATIONS, _LEGACY_GRAIN_ACTIVATION)

# AdamW and Muon are provided by PyTorch. The remaining published baselines
# are imported lazily from training.baseline_optimizers.
BASELINE_OPTIMIZERS = {
    "adamw",
    "muon",
    "lion",
    "ademamix",
    "schedule_free_adamw",
    "adafactor_came",
    "soap_adamw",
}
RLB_MATRIX_SYNC_OPTIMIZERS = set()
RLB_COEFFICIENT_SYNC_OPTIMIZERS = set()
RATIONAL_SPECIFIC_OPTIMIZERS = set()
ACTIVE_OPTIMIZERS = sorted(BASELINE_OPTIMIZERS | RATIONAL_SPECIFIC_OPTIMIZERS)
GRAIN_ACTIVATION_IDS = {GRAIN_ACTIVATION, _LEGACY_GRAIN_ACTIVATION}


def resolve_group_count(hidden_dim, group_size, max_groups):
    target = min(int(max_groups), max(1, math.ceil(int(hidden_dim) / int(group_size))))
    target = min(target, int(hidden_dim))
    for groups in range(target, 0, -1):
        if hidden_dim % groups == 0:
            return groups
    return 1


def grain_settings(ffn_dim, group_size, max_groups):
    hidden_dim = (3 * int(ffn_dim)) // 2
    return {
        "hidden_dim": hidden_dim,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "base_init": "silu",
        "centers": (),
        "coeff_limit": 0.0,
        "beta": 0.0,
        "input_affine": False,
        "center_odd": False,
        "train_centers": False,
        "atom_scale_init": None,
        "atom_scale_limit": 1.0,
        "fused": True,
    }


class SwiGLU(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.activation_name = "silu"
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.value = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.value(x))


class GRAINMLP(nn.Module):
    """Two-matrix MLP using Groupwise Rational Activation with Internal Normalization."""

    def __init__(self, dim, ffn_dim, group_size, max_groups, eps=1e-6):
        super().__init__()
        settings = grain_settings(ffn_dim, group_size, max_groups)
        self.activation_name = GRAIN_ACTIVATION
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)

        from rational_opt import GRAIN

        self.rlb_activation = GRAIN(
            self.hidden_dim,
            self.groups,
            init=settings["base_init"],
            fit_range=5.0,
            eps=eps,
        )
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.out_proj(self.rlb_activation(self.in_proj(x)))


def apply_rlb_positive_gauge(model: nn.Module, log_scale: float, seed: int) -> int:
    """Apply the optional positive A/B rescaling used by the fairness audit."""

    if log_scale <= 0.0:
        return 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    group_count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, GRAINMLP):
                continue
            groups = int(module.groups)
            width = int(module.hidden_dim // module.groups)
            logs = torch.empty(groups, dtype=torch.float32).uniform_(
                -float(log_scale), float(log_scale), generator=generator
            )
            scales = torch.exp(logs).to(
                device=module.in_proj.weight.device,
                dtype=module.in_proj.weight.dtype,
            )
            module.in_proj.weight.view(groups, width, -1).mul_(
                scales.view(groups, 1, 1)
            )
            module.out_proj.weight.view(
                module.out_proj.weight.shape[0], groups, width
            ).mul_(scales.reciprocal().view(1, groups, 1))
            group_count += groups
    return group_count


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ffn_dim,
        seq_len,
        activation,
        rational_init,
        rational_group_size,
        rational_max_groups,
        rational_eps,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, seq_len)
        if activation == "silu":
            self.mlp = SwiGLU(dim, ffn_dim)
        elif activation in GRAIN_ACTIVATION_IDS:
            if rational_init != "silu":
                raise ValueError("GRAIN uses the published SiLU initialization")
            self.mlp = GRAINMLP(
                dim,
                ffn_dim,
                rational_group_size,
                rational_max_groups,
                rational_eps,
            )
        else:
            raise ValueError(f"unknown activation {activation!r}")

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, args.d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    args.d_model,
                    args.heads,
                    args.ffn_dim,
                    args.seq_len,
                    args.activation,
                    args.rational_init,
                    args.rational_group_size,
                    args.rational_max_groups,
                    args.rational_basis_eps,
                )
                for _ in range(args.layers)
            ]
        )
        self.norm = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(lambda module: init_weights(module, args.init_std))

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

def init_weights(module, init_std):
    if isinstance(module, nn.Linear):
        if getattr(module, "zero_init", False):
            nn.init.zeros_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=init_std)


def setup_distributed():
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size, torch.device("cuda", local_rank)


def cleanup_distributed(is_distributed):
    if is_distributed:
        dist.destroy_process_group()


def rank0_print(rank, message):
    if rank == 0:
        print(message, flush=True)


def sanitize_name(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def normalize_dataset_config(config):
    if config is None:
        return None
    if str(config).lower() in {"", "none", "null"}:
        return None
    return config


DOLMA_MANIFESTS = {
    "v1_5-sample": "urls/v1_5-sample.txt",
    "v1_6-sample": "urls/v1_6-sample.txt",
    "v1_6": "urls/v1_6.txt",
}


def actual_dataset_split(args, split):
    if split == "train":
        return args.train_split
    if split == "validation":
        return args.validation_split
    return split


def dataset_skip_documents(args, split):
    if split == "train":
        return max(0, int(args.train_skip_documents))
    if split == "validation":
        return max(0, int(args.validation_skip_documents))
    return 0


def dataset_skip_tokens(args, split):
    if split == "train":
        return max(0, int(args.train_skip_tokens))
    if split == "validation":
        return max(0, int(args.validation_skip_tokens))
    return 0


def append_tokenized_ids(tokens, ids, eos_id, max_tokens, skip_tokens):
    if not ids:
        return skip_tokens, False
    doc = list(ids)
    doc.append(eos_id)
    if skip_tokens > 0:
        if skip_tokens >= len(doc):
            return skip_tokens - len(doc), False
        doc = doc[skip_tokens:]
        skip_tokens = 0
    tokens.extend(doc)
    return skip_tokens, max_tokens is not None and len(tokens) >= max_tokens


def token_cache_path(args, split, max_tokens):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_part = "all" if max_tokens is None else str(max_tokens)
    config_name = args.dataset_config if args.dataset_config else "none"
    actual_split = actual_dataset_split(args, split)
    skip_docs = dataset_skip_documents(args, split)
    skip_tokens = dataset_skip_tokens(args, split)
    revision_part = ""
    if args.dataset_revision is not None or args.tokenizer_revision is not None:
        dataset_revision = args.dataset_revision or "default"
        tokenizer_revision = args.tokenizer_revision or "default"
        revision_part = (
            f"_dsrev{sanitize_name(dataset_revision)}"
            f"_tokrev{sanitize_name(tokenizer_revision)}"
        )
    legacy_cache_name = (
        not args.dataset_streaming
        and args.dataset_text_column == "text"
        and actual_split == split
        and skip_docs == 0
        and skip_tokens == 0
        and args.train_split == "train"
        and args.validation_split == "validation"
        and not revision_part
    )
    if legacy_cache_name:
        name = (
            f"{sanitize_name(args.dataset_name)}_{sanitize_name(config_name)}_"
            f"{sanitize_name(args.tokenizer)}_{split}_{max_part}.pt"
        )
    else:
        stream_part = "stream" if args.dataset_streaming else "map"
        name = (
            f"{sanitize_name(args.dataset_name)}_{sanitize_name(config_name)}_"
            f"{sanitize_name(args.tokenizer)}_{sanitize_name(actual_split)}_{split}_"
            f"{stream_part}_{sanitize_name(args.dataset_text_column)}_skipdocs{skip_docs}_"
            f"skiptoks{skip_tokens}{revision_part}_{max_part}.pt"
        )
    return cache_dir / name


def extract_text_from_record(record, text_column):
    if text_column == "auto":
        for candidate in ("text", "content", "document", "raw_content"):
            value = record.get(candidate)
            if value:
                return value
        return None
    value = record
    for part in text_column.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def extract_texts_from_batch(batch, text_column):
    if text_column != "auto" and text_column in batch:
        values = batch[text_column]
    else:
        values = None
        for candidate in ("text", "content", "document", "raw_content"):
            if candidate in batch:
                values = batch[candidate]
                break
    if values is None:
        available = ", ".join(sorted(batch.keys()))
        raise KeyError(f"could not find text column {text_column!r}; available columns: {available}")
    return [text for text in values if isinstance(text, str) and text.strip()]


def load_dolma_manifest_dataset(args, dataset_config, actual_split):
    if actual_split != "train":
        raise ValueError(
            "allenai/dolma manifest loading supports train split only; "
            "use train plus skip offsets for validation"
        )
    manifest = DOLMA_MANIFESTS.get(dataset_config)
    if manifest is None:
        known = ", ".join(sorted(DOLMA_MANIFESTS))
        raise ValueError(
            f"unsupported allenai/dolma config {dataset_config!r}; "
            f"expected one of: {known}"
        )
    manifest_path = hf_hub_download(
        repo_id="allenai/dolma",
        filename=manifest,
        repo_type="dataset",
        cache_dir=args.hf_cache,
        revision=args.dataset_revision,
    )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        urls = [line.strip() for line in handle if line.strip()]
    if not urls:
        raise ValueError(f"empty allenai/dolma URL manifest: {manifest}")
    return load_dataset(
        "json",
        data_files={"train": urls},
        split="train",
        cache_dir=args.hf_cache,
        streaming=args.dataset_streaming,
    )


def load_hf_dataset(args, split):
    dataset_config = normalize_dataset_config(args.dataset_config)
    actual_split = actual_dataset_split(args, split)
    if args.dataset_name == "allenai/dolma" and dataset_config in DOLMA_MANIFESTS:
        return load_dolma_manifest_dataset(args, dataset_config, actual_split)
    kwargs = {
        "split": actual_split,
        "cache_dir": args.hf_cache,
        "streaming": args.dataset_streaming,
        "revision": args.dataset_revision,
    }
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if dataset_config is None:
        return load_dataset(args.dataset_name, **kwargs)
    return load_dataset(args.dataset_name, dataset_config, **kwargs)


def synthetic_arithmetic_text(index: int) -> str:
    a = (index * 48271 + 17) % 10000
    b = (index * 69621 + 23) % 10000
    c = a + b
    lo = min(a, b)
    hi = max(a, b)
    step = index % 17 + 2
    first = (index * 37 + 11) % 2000
    seq = [first + step * offset for offset in range(5)]
    mode = index % 4
    if mode == 0:
        return f"Task add: {a} plus {b} equals {c}. Check: {c} minus {b} equals {a}."
    if mode == 1:
        return f"Task compare: between {a} and {b}, smaller is {lo}, larger is {hi}. Difference is {hi - lo}."
    if mode == 2:
        return f"Task sequence: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}; next is {seq[4]}. Step size is {step}."
    product = (a % 100) * (b % 100)
    return f"Task multiply-small: {a % 100} times {b % 100} equals {product}. Inputs came from {a} and {b}."


def synthetic_code_text(index: int) -> str:
    base = (index * 1103515245 + 12345) & 0x7FFFFFFF
    x = base % 97
    y = (base // 97) % 89
    z = (3 * x + 2 * y + index) % 211
    name = f"v{index % 17}"
    mode = index % 5
    if mode == 0:
        return f"def f({name}, y):\n    total = {name} + {x}\n    total = total * {y % 7 + 2}\n    return total\n# f({y}, {z % 11}) -> {(y + x) * (y % 7 + 2)}"
    if mode == 1:
        values = [(x + k * (y % 5 + 1)) % 100 for k in range(5)]
        return f"items = {values}\nacc = 0\nfor item in items:\n    acc += item\nprint(acc)  # {sum(values)}"
    if mode == 2:
        flag = (x + y) % 2 == 0
        chosen = x if flag else y
        return f"left = {x}\nright = {y}\nanswer = left if (left + right) % 2 == 0 else right\nanswer == {chosen}"
    if mode == 3:
        keys = [f"k{(index + j) % 9}" for j in range(3)]
        vals = [(x + y + j * 13) % 50 for j in range(3)]
        return f"data = {{{keys[0]!r}: {vals[0]}, {keys[1]!r}: {vals[1]}, {keys[2]!r}: {vals[2]}}}\nlookup = {keys[1]!r}\nvalue = data[lookup]  # {vals[1]}"
    final = x + 6 if x % 2 == (x + 6) % 2 else x + 7
    return f"while {name} < {x + 6}:\n    {name} += 2\n# start {x}, stop {x + 6}, final {final}"


def synthetic_symbolic_text(index: int) -> str:
    a = chr(ord("A") + index % 8)
    b = chr(ord("A") + (index * 3 + 1) % 8)
    c = chr(ord("A") + (index * 5 + 2) % 8)
    n = index % 7 + 3
    mode = index % 5
    if mode == 0:
        return f"Rewrite rule: ({a} -> {b}) and ({b} -> {c}). Query {a}. Result {c}."
    if mode == 1:
        seq = [((index + 2) * (k + 1) + n) % 19 for k in range(6)]
        return f"Map every token by +{n}: input {seq[:4]} output {[value + n for value in seq[:4]]}."
    if mode == 2:
        bits = [(index >> k) & 1 for k in range(6)]
        parity = sum(bits) % 2
        return f"Boolean trace: bits {bits}; xor parity is {parity}; not parity is {1 - parity}."
    if mode == 3:
        opens = index % 4 + 1
        token = "(" * opens + a + ")" * opens
        return f"Bracket task: source {token}; depth {opens}; core token {a}."
    left = [a, b, c, a]
    right = list(reversed(left))
    return f"Reverse-copy task: source {' '.join(left)}; target {' '.join(right)}."


def synthetic_reasoning_text(index: int) -> str:
    mode = index % 3
    if mode == 0:
        return synthetic_arithmetic_text(index)
    if mode == 1:
        return synthetic_code_text(index)
    return synthetic_symbolic_text(index)


def synthetic_text_for_dataset(dataset_name: str, index: int) -> str:
    if dataset_name == "synthetic/arithmetic":
        return synthetic_arithmetic_text(index)
    if dataset_name == "synthetic/code":
        return synthetic_code_text(index)
    if dataset_name == "synthetic/symbolic":
        return synthetic_symbolic_text(index)
    if dataset_name == "synthetic/reasoning_mix":
        return synthetic_reasoning_text(index)
    allowed = "synthetic/arithmetic, synthetic/code, synthetic/symbolic, synthetic/reasoning_mix"
    raise ValueError(f"unknown synthetic dataset {dataset_name!r}; expected one of: {allowed}")


def tokenize_synthetic_text(args, split, max_tokens, tokenizer, eos_id):
    if max_tokens is None:
        raise ValueError(f"{args.dataset_name} requires max_tokens")
    tokens = array("I")
    split_offset = {"train": 0, "validation": 50_000_000, "test": 75_000_000}.get(split, 90_000_000)
    index = 0
    while len(tokens) < max_tokens:
        texts = [
            synthetic_text_for_dataset(args.dataset_name, split_offset + index + item)
            for item in range(args.tokenize_batch_size)
        ]
        index += args.tokenize_batch_size
        batch = tokenizer(texts, add_special_tokens=False)
        for ids in batch["input_ids"]:
            if ids:
                tokens.extend(ids)
                tokens.append(eos_id)
            if len(tokens) >= max_tokens:
                break
    return tokens


def _cache_metadata(args, split):
    return {
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "split": split,
        "actual_split": actual_dataset_split(args, split),
        "dataset_streaming": args.dataset_streaming,
        "dataset_text_column": args.dataset_text_column,
        "skip_documents": dataset_skip_documents(args, split),
        "skip_tokens": dataset_skip_tokens(args, split),
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
    }


def _load_cached_tokens(cache_file, args, split, max_tokens):
    payload = torch.load(cache_file, map_location="cpu")
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("tokens")):
        raise RuntimeError(f"invalid token cache payload: {cache_file}")
    mismatches = {
        key: {"observed": payload.get(key), "required": required}
        for key, required in _cache_metadata(args, split).items()
        if payload.get(key) != required
    }
    tokens = payload["tokens"]
    if payload.get("schema") != "rationalopt_token_cache_v2":
        mismatches["schema"] = {
            "observed": payload.get("schema"),
            "required": "rationalopt_token_cache_v2",
        }
    if tokens.dtype != torch.int32:
        mismatches["dtype"] = {
            "observed": str(tokens.dtype),
            "required": str(torch.int32),
        }
    if payload.get("token_count") != int(tokens.numel()):
        mismatches["payload_token_count"] = {
            "observed": payload.get("token_count"),
            "required": int(tokens.numel()),
        }
    content_sha256 = payload.get("token_content_sha256")
    if (
        not isinstance(content_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None
    ):
        mismatches["token_content_sha256"] = {
            "observed": content_sha256,
            "required": "64 lowercase hexadecimal characters",
        }
    if max_tokens is not None and int(tokens.numel()) != int(max_tokens):
        mismatches["token_count"] = {
            "observed": int(tokens.numel()),
            "required": int(max_tokens),
        }
    if int(tokens.numel()) < int(args.seq_len) + 2:
        mismatches["minimum_token_count"] = {
            "observed": int(tokens.numel()),
            "required": int(args.seq_len) + 2,
        }
    if not tokens.is_contiguous():
        mismatches["contiguous"] = {"observed": False, "required": True}
    if mismatches:
        raise RuntimeError(f"token cache metadata mismatch for {cache_file}: {mismatches}")
    return tokens


def load_or_tokenize(args, split, max_tokens):
    cache_file = token_cache_path(args, split, max_tokens)
    lock_file = Path(f"{cache_file}.lock")
    with lock_file.open("a+b") as lock_handle:
        if not args.refresh_cache:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            if cache_file.exists():
                return _load_cached_tokens(
                    cache_file, args, split, max_tokens
                )
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_file.exists() and not args.refresh_cache:
            return _load_cached_tokens(cache_file, args, split, max_tokens)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer,
            cache_dir=args.hf_cache,
            revision=args.tokenizer_revision,
        )
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = tokenizer.pad_token_id
        if eos_id is None:
            raise ValueError("tokenizer must define eos_token_id or pad_token_id")

        if args.dataset_name.startswith("synthetic/"):
            tokens = tokenize_synthetic_text(
                args, split, max_tokens, tokenizer, eos_id
            )
        else:
            dataset = load_hf_dataset(args, split)
            skip_docs = dataset_skip_documents(args, split)
            skip_tokens = dataset_skip_tokens(args, split)
            tokens = array("I")
            if args.dataset_streaming:
                if skip_docs > 0 and hasattr(dataset, "skip"):
                    dataset = dataset.skip(skip_docs)
                    skipped = skip_docs
                else:
                    skipped = 0
                texts = []
                for record in dataset:
                    if skipped < skip_docs:
                        skipped += 1
                        continue
                    text = extract_text_from_record(
                        record, args.dataset_text_column
                    )
                    if isinstance(text, str) and text.strip():
                        texts.append(text)
                    if len(texts) < args.tokenize_batch_size:
                        continue
                    batch = tokenizer(texts, add_special_tokens=False)
                    texts.clear()
                    for ids in batch["input_ids"]:
                        skip_tokens, done = append_tokenized_ids(
                            tokens, ids, eos_id, max_tokens, skip_tokens
                        )
                        if done:
                            break
                    if max_tokens is not None and len(tokens) >= max_tokens:
                        break
                if texts and (max_tokens is None or len(tokens) < max_tokens):
                    batch = tokenizer(texts, add_special_tokens=False)
                    for ids in batch["input_ids"]:
                        skip_tokens, done = append_tokenized_ids(
                            tokens, ids, eos_id, max_tokens, skip_tokens
                        )
                        if done:
                            break
            else:
                for start in range(
                    skip_docs, len(dataset), args.tokenize_batch_size
                ):
                    end = min(
                        start + args.tokenize_batch_size, len(dataset)
                    )
                    texts = extract_texts_from_batch(
                        dataset[start:end], args.dataset_text_column
                    )
                    if not texts:
                        continue
                    batch = tokenizer(texts, add_special_tokens=False)
                    for ids in batch["input_ids"]:
                        skip_tokens, done = append_tokenized_ids(
                            tokens, ids, eos_id, max_tokens, skip_tokens
                        )
                        if done:
                            break
                    if max_tokens is not None and len(tokens) >= max_tokens:
                        break

        np_tokens = np.frombuffer(tokens, dtype=np.uint32)
        if max_tokens is not None:
            np_tokens = np_tokens[:max_tokens]
        if np_tokens.size < args.seq_len + 2:
            raise ValueError(f"{split} split produced too few tokens")
        if int(np_tokens.max()) > np.iinfo(np.int32).max:
            raise ValueError("token ids exceed int32 range")
        tensor = torch.from_numpy(np_tokens.astype(np.int32, copy=True))
        payload = {
            "schema": "rationalopt_token_cache_v2",
            "tokens": tensor,
            "token_count": int(tensor.numel()),
            "token_content_sha256": tensor_content_sha256(tensor),
            **_cache_metadata(args, split),
        }
        temporary = cache_file.with_name(
            f".{cache_file.name}.tmp-{os.getpid()}"
        )
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, cache_file)
            directory_fd = os.open(cache_file.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
        return _load_cached_tokens(cache_file, args, split, max_tokens)


def sample_batch(tokens, batch_size, seq_len, offsets, generator, device):
    max_start = tokens.numel() - seq_len - 1
    starts = torch.randint(max_start, (batch_size,), generator=generator)
    block = tokens[starts[:, None] + offsets[None, :]]
    x = block[:, :-1].to(device=device, dtype=torch.long, non_blocking=True)
    y = block[:, 1:].to(device=device, dtype=torch.long, non_blocking=True)
    return x, y


def tensor_content_sha256(tensor, chunk_bytes=64 * 1024 * 1024):
    """Hash a contiguous tensor's logical bytes without one large byte copy."""

    if not torch.is_tensor(tensor):
        raise TypeError("tensor_content_sha256 expects a tensor")
    if int(chunk_bytes) <= 0:
        raise ValueError("chunk_bytes must be positive")
    value = tensor.detach().cpu().contiguous()
    byte_view = value.view(torch.uint8).reshape(-1).numpy()
    memory = memoryview(byte_view)
    digest = hashlib.sha256()
    for start in range(0, memory.nbytes, int(chunk_bytes)):
        digest.update(memory[start : start + int(chunk_bytes)])
    return digest.hexdigest()


def tensor_sample_sha256(tensor, sample_count=4096):
    values = tensor.detach().reshape(-1).cpu()
    count = min(int(sample_count), int(values.numel()))
    digest = hashlib.sha256()
    digest.update(str(tuple(values.shape)).encode("ascii"))
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(int(values.numel())).encode("ascii"))
    if count > 0:
        digest.update(values[:count].contiguous().numpy().tobytes())
        digest.update(values[-count:].contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sampled_start_fingerprint(token_count, seq_len, batch_size, draws, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    starts = torch.randint(
        int(token_count) - int(seq_len) - 1,
        (int(draws), int(batch_size)),
        generator=generator,
    )
    return tensor_sample_sha256(starts, sample_count=starts.numel())


def nvidia_smi_metadata(strict=False):
    fields = [
        "index",
        "uuid",
        "name",
        "pci.bus_id",
        "clocks.max.sm",
        "power.limit",
        "pstate",
        "utilization.gpu",
        "memory.used",
        "driver_version",
    ]
    try:
        command = ["nvidia-smi"]
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            command.extend(["-i", visible_devices])
        command.extend([f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        rows = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            values = [value.strip() for value in line.split(",")]
            if len(values) != len(fields):
                raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
            rows.append(dict(zip(fields, values)))
        if strict and len(rows) != 4:
            raise RuntimeError(f"expected four visible GPUs, found {len(rows)}")
        if strict and any("A6000" not in row["name"] for row in rows):
            raise RuntimeError(f"expected only A6000 GPUs, found {[row['name'] for row in rows]}")
        return rows
    except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError):
        if strict:
            raise
        return None


def reduce_mean(value, device, is_distributed):
    tensor = torch.tensor(float(value), device=device)
    if is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def reduce_max(value, device, is_distributed):
    tensor = torch.tensor(float(value), device=device)
    if is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def unwrap_model(model):
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def _finite_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _tensor_mean_std(tensor):
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return None, None
    values = tensor.detach().float().reshape(-1)
    return _finite_float(values.mean().item()), _finite_float(values.std(unbiased=False).item() if values.numel() > 1 else 0.0)


def _tensor_quantiles(tensor):
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return None, None, None
    values = tensor.detach().float().reshape(-1).cpu()
    return (
        _finite_float(values.min().item()),
        _finite_float(torch.quantile(values, 0.01).item()),
        _finite_float(values.median().item()),
    )


def grad_global_norm(model):
    total = None
    for param in model.parameters():
        if param.grad is None:
            continue
        value = param.grad.detach().float().square().sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(torch.sqrt(total).item())


def clip_or_measure_gradients(model, grad_clip, capture_norm):
    if grad_clip > 0:
        norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        norm_value = float(norm.item() if torch.is_tensor(norm) else norm)
        return norm_value, bool(norm_value > float(grad_clip))
    if capture_norm:
        return grad_global_norm(model), False
    return None, False


@torch.no_grad()
def assert_rlb_matrix_parameters_synced(model, args, device, is_distributed):
    """Require exact equality of every GRAIN A/B element across DDP ranks."""

    parameters = []
    for group in collect_rlb_optimizer_groups(unwrap_model(model), args):
        parameters.extend((group["in_weight"], group["out_weight"]))
    if not parameters:
        raise RuntimeError("DDP parameter synchronization check found no GRAIN matrix pair")
    if not is_distributed:
        return 0.0
    world_size = dist.get_world_size()
    maximum = 0.0
    chunk_elements = 1_048_576
    for parameter in parameters:
        flat = parameter.detach().float().reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            local = flat[start : start + chunk_elements].to(device=device)
            gathered = [torch.empty_like(local) for _ in range(world_size)]
            dist.all_gather(gathered, local)
            reference = gathered[0]
            for peer in gathered[1:]:
                difference = float((peer - reference).abs().max().item())
                maximum = max(maximum, difference)
                if difference != 0.0:
                    raise RuntimeError(
                        "DDP GRAIN matrix replicas diverged after optimizer step; "
                        f"max absolute difference={difference}"
                    )
    return maximum


@torch.no_grad()
def assert_rlb_coefficient_parameters_synced(model, args, device, is_distributed):
    """Require exact DDP equality of every GRAIN numerator/denominator value."""

    coefficient_values = []
    for group in collect_rlb_optimizer_groups(unwrap_model(model), args):
        coefficient_values.extend(
            (
                group["numerator"].detach().float().reshape(-1),
                group["denominator"].detach().float().reshape(-1),
            )
        )
    if not coefficient_values:
        raise RuntimeError("DDP coefficient synchronization check found no GRAIN coefficients")
    if not is_distributed:
        return 0.0
    local = torch.cat(coefficient_values).to(device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    reference = gathered[0]
    max_abs = max(float((peer - reference).abs().max().item()) for peer in gathered[1:])
    if max_abs != 0.0:
        raise RuntimeError(
            f"DDP GRAIN coefficient replicas diverged after optimizer step; max absolute difference={max_abs}"
        )
    return max_abs


def iter_optimizer_tree(optimizer):
    yield optimizer
    for child in getattr(optimizer, "optimizers", []):
        yield from iter_optimizer_tree(child)
    for attr in ("adam", "muon"):
        child = getattr(optimizer, attr, None)
        if child is not None:
            yield child


def set_optimizer_telemetry_capture(optimizer, enabled):
    for item in iter_optimizer_tree(optimizer):
        setter = getattr(item, "set_telemetry_capture", None)
        if setter is not None:
            setter(enabled)


def collect_optimizer_telemetry(optimizer):
    record = {}
    for item in iter_optimizer_tree(optimizer):
        getter = getattr(item, "telemetry", None)
        if getter is None:
            continue
        telemetry = getter()
        if telemetry:
            record.update(telemetry)
    return record


def parse_optimizer_telemetry_steps(value, total_steps):
    """Optional audit-only optimizer telemetry schedule; default is unchanged."""

    if value in (None, ""):
        return None
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*(?:,[1-9][0-9]*)*", value) is None:
        raise RuntimeError("TILLER_OPTIMIZER_TELEMETRY_STEPS must be ascending comma-separated positive steps")
    steps = tuple(int(item) for item in value.split(","))
    if tuple(sorted(set(steps))) != steps or steps[-1] > int(total_steps):
        raise RuntimeError("TILLER_OPTIMIZER_TELEMETRY_STEPS is duplicated, unordered, or beyond the run")
    return steps


def enable_rlb_training_telemetry(model, args):
    if args.activation not in GRAIN_ACTIVATION_IDS:
        return
    for module in model.modules():
        if not all(hasattr(module, attr) for attr in ("groups", "hidden_dim", "numerator", "denominator")):
            continue
        setattr(module, "_rlb_optimizer_track_stats", True)
        setattr(module, "_rlb_optimizer_stat_every", max(1, int(args.telemetry_rlb_stat_every)))
        setattr(module, "_rlb_optimizer_stat_samples", max(1, int(args.telemetry_rlb_stat_samples)))


def rlb_live_stats_scope(model):
    tracked = False
    optimizer_consumed = False
    for module in unwrap_model(model).modules():
        if not bool(getattr(module, "_rlb_optimizer_track_stats", False)):
            continue
        tracked = True
        optimizer_consumed = optimizer_consumed or bool(
            getattr(module, "_rlb_optimizer_sync_stats", False)
        )
    if optimizer_consumed:
        return "optimizer_gains_global_weighted_train_only"
    if tracked:
        return "telemetry_only"
    return "disabled"


def _rlb_denominator_probe(group, points=129, probe_range=5.0):
    module = group.get("module")
    if module is None:
        return None
    denominator = getattr(module, "denominator", None)
    centers = getattr(module, "centers", None)
    beta = getattr(module, "beta", None)
    if denominator is None:
        return None
    device = denominator.device
    dtype = torch.float32
    t = torch.linspace(-float(probe_range), float(probe_range), int(points), device=device, dtype=dtype)
    abs_t = t.abs()
    t2 = t.square()
    t3_abs = abs_t * t2
    t4 = t2.square()
    den = denominator.detach().float().abs()
    if den.dim() != 2 or den.size(-1) < 4:
        return None
    base_q = 1.0 + den[:, 0:1] * abs_t + den[:, 1:2] * t2 + den[:, 2:3] * t3_abs + den[:, 3:4] * t4
    values = [base_q.reshape(-1)]
    if centers is not None and beta is not None:
        center = centers.detach().float().unsqueeze(-1)
        beta_v = beta.detach().float().unsqueeze(-1).clamp_min(0.0)
        local_q = 1.0 + beta_v * (t.view(1, 1, -1) - center).square()
        values.append(local_q.reshape(-1))
    return torch.cat(values)


def collect_rlb_telemetry(model, args):
    if args.activation not in GRAIN_ACTIVATION_IDS:
        return {}
    groups = collect_rlb_optimizer_groups(unwrap_model(model), args)
    if not groups:
        return {}

    result = {}
    output_means, output_stds = [], []
    derivative_means, derivative_stds = [], []
    atom_means, atom_stds = [], []
    abs_moment_means, abs_moment_stds = [], []
    denom_mins, denom_p01s, denom_medians = [], [], []
    w_in_means, w_in_stds = [], []
    w_out_means, w_out_stds = [], []
    log_ratio_means, log_product_means = [], []

    for group in groups:
        module = group.get("module")
        stats = getattr(module, "_rlb_optimizer_stats", None) if module is not None else None
        for key, means, stds in (
            ("output_rms", output_means, output_stds),
            ("derivative_rms", derivative_means, derivative_stds),
            ("atom_rms", atom_means, atom_stds),
            ("abs_moments", abs_moment_means, abs_moment_stds),
        ):
            mean, std = _tensor_mean_std(None if not stats else stats.get(key))
            means.append(mean)
            stds.append(std)

        den_values = _rlb_denominator_probe(group, args.telemetry_denominator_probe_points)
        den_min, den_p01, den_median = _tensor_quantiles(den_values)
        denom_mins.append(den_min)
        denom_p01s.append(den_p01)
        denom_medians.append(den_median)

        in_weight = group["in_weight"].detach().float()
        out_weight = group["out_weight"].detach().float()
        groups_count = int(group["groups"])
        hidden_dim = int(group["hidden_dim"])
        width = hidden_dim // groups_count
        in_view = in_weight.view(groups_count, width, -1)
        out_view = out_weight.view(out_weight.shape[0], groups_count, width).permute(1, 2, 0)
        in_rms = torch.sqrt(in_view.square().mean(dim=(1, 2)) + 1e-12)
        out_rms = torch.sqrt(out_view.square().mean(dim=(1, 2)) + 1e-12)
        in_mean, in_std = _tensor_mean_std(in_rms)
        out_mean, out_std = _tensor_mean_std(out_rms)
        w_in_means.append(in_mean)
        w_in_stds.append(in_std)
        w_out_means.append(out_mean)
        w_out_stds.append(out_std)
        log_ratio_means.append(_finite_float((torch.log(in_rms) - torch.log(out_rms)).mean().item()))
        log_product_means.append(_finite_float((torch.log(in_rms) + torch.log(out_rms)).mean().item()))

    result.update(
        {
            "grain_output_rms_mean_by_layer": output_means,
            "grain_output_rms_std_by_layer": output_stds,
            "grain_derivative_rms_mean_by_layer": derivative_means,
            "grain_derivative_rms_std_by_layer": derivative_stds,
            "grain_atom_rms_mean_by_layer": atom_means,
            "grain_atom_rms_std_by_layer": atom_stds,
            "grain_abs_moment_mean_by_layer": abs_moment_means,
            "grain_abs_moment_std_by_layer": abs_moment_stds,
            "denominator_abs_min_by_layer": denom_mins,
            "denominator_abs_p01_by_layer": denom_p01s,
            "denominator_abs_median_by_layer": denom_medians,
            "w_in_rms_mean_by_layer": w_in_means,
            "w_in_rms_std_by_layer": w_in_stds,
            "w_out_rms_mean_by_layer": w_out_means,
            "w_out_rms_std_by_layer": w_out_stds,
            "log_w_in_over_w_out_by_layer": log_ratio_means,
            "log_norm_product_by_layer": log_product_means,
        }
    )
    return result


def svd_entropy(matrix, max_dim):
    weight = matrix.detach().float()
    if weight.dim() != 2 or min(weight.shape) <= 1:
        return None
    rows, cols = weight.shape
    if max_dim > 0 and rows > max_dim:
        row_index = torch.linspace(0, rows - 1, max_dim, device=weight.device).long()
        weight = weight.index_select(0, row_index)
    if max_dim > 0 and cols > max_dim:
        col_index = torch.linspace(0, cols - 1, max_dim, device=weight.device).long()
        weight = weight.index_select(1, col_index)
    try:
        singular = torch.linalg.svdvals(weight)
    except RuntimeError:
        return None
    singular = singular.clamp_min(0.0)
    total = singular.sum()
    if not torch.isfinite(total) or total <= 0:
        return None
    probs = singular / total
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    norm = math.log(max(2, probs.numel()))
    return _finite_float((entropy / norm).item())


def collect_matrix_spectrum_telemetry(model, args):
    if args.matrix_spectrum_interval <= 0:
        return {}
    raw_model = unwrap_model(model)
    values = {
        "svd_entropy_attn_q": [],
        "svd_entropy_attn_k": [],
        "svd_entropy_attn_v": [],
        "svd_entropy_attn_o": [],
        "svd_entropy_grain_in": [],
        "svd_entropy_grain_out": [],
    }
    for layer in getattr(raw_model, "layers", []):
        attn = getattr(layer, "attn", None)
        if attn is not None and hasattr(attn, "qkv"):
            q_weight, k_weight, v_weight = attn.qkv.weight.chunk(3, dim=0)
            for key, weight in (("svd_entropy_attn_q", q_weight), ("svd_entropy_attn_k", k_weight), ("svd_entropy_attn_v", v_weight)):
                value = svd_entropy(weight, args.matrix_spectrum_max_dim)
                if value is not None:
                    values[key].append(value)
        if attn is not None and hasattr(attn, "out"):
            value = svd_entropy(attn.out.weight, args.matrix_spectrum_max_dim)
            if value is not None:
                values["svd_entropy_attn_o"].append(value)
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, GRAINMLP):
            value = svd_entropy(mlp.in_proj.weight, args.matrix_spectrum_max_dim)
            if value is not None:
                values["svd_entropy_grain_in"].append(value)
            value = svd_entropy(mlp.out_proj.weight, args.matrix_spectrum_max_dim)
            if value is not None:
                values["svd_entropy_grain_out"].append(value)
    return {key: _finite_float(sum(items) / len(items)) for key, items in values.items() if items}


def prepare_probe_batch(tokens, args, offsets, rank, device, out_path):
    if args.probe_batch_size <= 0:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 2_000_003 + rank)
    batch_size = min(int(args.probe_batch_size), int(args.batch_size))
    x, y = sample_batch(tokens, batch_size, args.seq_len, offsets, generator, device)
    probe_path = out_path.parent / f"{sanitize_name(args.activation)}_probe_rank{rank}.pt"
    if rank == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    torch.save(
        {
            "probe_x": x.detach().cpu(),
            "probe_y": y.detach().cpu(),
            "dataset": args.dataset_name,
            "dataset_config": args.dataset_config,
            "validation_skip_tokens": args.validation_skip_tokens,
            "seed": args.seed,
            "rank": rank,
        },
        probe_path,
    )
    return {"x": x, "prev_logits": None, "first_logits": None}


@torch.no_grad()
def evaluate_probe(model, probe_state, device, is_distributed):
    if probe_state is None:
        return {}
    model.eval()
    logits = model(probe_state["x"]).detach().float()
    logit_rms_local = torch.sqrt(logits.square().mean() + 1e-12)
    metrics = {"probe_logit_rms": reduce_mean(float(logit_rms_local.item()), device, is_distributed)}

    for label, reference in (("since_prev_eval", probe_state.get("prev_logits")), ("since_step1", probe_state.get("first_logits"))):
        if reference is None:
            metrics[f"probe_logit_delta_rms_{label}"] = 0.0
            metrics[f"probe_kl_{label}"] = 0.0
            continue
        ref = reference.to(device=device, dtype=torch.float32)
        delta_rms = torch.sqrt((logits - ref).square().mean() + 1e-12)
        log_probs = F.log_softmax(logits, dim=-1)
        ref_log_probs = F.log_softmax(ref, dim=-1)
        kl = (ref_log_probs.exp() * (ref_log_probs - log_probs)).sum(dim=-1).mean()
        metrics[f"probe_logit_delta_rms_{label}"] = reduce_mean(float(delta_rms.item()), device, is_distributed)
        metrics[f"probe_kl_{label}"] = reduce_mean(float(kl.item()), device, is_distributed)

    stored = logits.detach().to(device="cpu", dtype=torch.float16)
    if probe_state.get("first_logits") is None:
        probe_state["first_logits"] = stored.clone()
    probe_state["prev_logits"] = stored
    return metrics


@torch.no_grad()
def evaluate(model, tokens, args, offsets, rank, world_size, device, is_distributed):
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1_000_003 + rank)
    total_loss = 0.0
    for _ in range(args.eval_batches):
        x, y = sample_batch(tokens, args.batch_size, args.seq_len, offsets, generator, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        total_loss += float(loss.item())
    local_mean = total_loss / max(1, args.eval_batches)
    return reduce_mean(local_mean, device, is_distributed)


class CompositeOptimizer:
    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        self.param_groups = []
        for optimizer in self.optimizers:
            self.param_groups.extend(optimizer.param_groups)

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dict):
        for optimizer, optimizer_state in zip(self.optimizers, state_dict):
            optimizer.load_state_dict(optimizer_state)


EXACT_LR_WD_CONTRACT = "exact_lr_wd_v1"
TILLER_LR_WD_CONTRACT = EXACT_LR_WD_CONTRACT
TILLER_EXPERIMENT_IDENTITY = "tiller_matrix_v1"
_TILLER_IDENTITY_AUDITOR = None


def install_tiller_identity_auditor(auditor):
    """Install the suite-specific model, data, and schedule identity check."""

    global _TILLER_IDENTITY_AUDITOR
    if not callable(auditor):
        raise TypeError("TILLER identity auditor must be callable")
    _TILLER_IDENTITY_AUDITOR = auditor


def audit_tiller_experiment_identity(
    args,
    *,
    world_size,
    global_tokens,
    train_token_count,
    val_token_count,
    parameter_count,
):
    """Evaluate the identity contract installed by the experiment suite."""

    if args.experiment_identity != TILLER_EXPERIMENT_IDENTITY:
        return None
    if _TILLER_IDENTITY_AUDITOR is None:
        raise RuntimeError("the TILLER suite has not installed its identity auditor")
    return _TILLER_IDENTITY_AUDITOR(
        args,
        world_size=world_size,
        global_tokens=global_tokens,
        train_token_count=train_token_count,
        val_token_count=val_token_count,
        parameter_count=parameter_count,
    )


def _optimizer_tree(optimizer):
    """Yield each optimizer object once, including composite children."""
    pending = [optimizer]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(getattr(current, "optimizers", ()))


def _unit_scalar(value, *, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be the literal scalar 1.0, got {value!r}")
    if float(value) != 1.0:
        raise RuntimeError(f"{label} must be 1.0, got {value!r}")


def audit_optimizer_lr_wd_fairness(model, optimizer, args):
    """Verify equal LR/WD scaling across outer groups and optimizer internals."""

    if args.fairness_contract != EXACT_LR_WD_CONTRACT:
        return None
    raw_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    named_parameters = {
        id(param): (name, param)
        for name, param in raw_model.named_parameters()
        if param.requires_grad
    }
    observed = {}
    group_records = []
    for group_index, group in enumerate(optimizer.param_groups):
        _unit_scalar(group.get("lr_scale", 1.0), label=f"param_groups[{group_index}].lr_scale")
        if float(group.get("lr", args.lr)) != float(args.lr):
            raise RuntimeError(
                f"param_groups[{group_index}].lr must equal base LR {args.lr}, "
                f"got {group.get('lr')!r}"
            )
        weight_decay = float(group.get("weight_decay", 0.0))
        names = []
        element_count = 0
        for param in group["params"]:
            if id(param) not in named_parameters:
                raise RuntimeError("optimizer contains an unknown or frozen parameter")
            name, original = named_parameters[id(param)]
            if id(param) in observed:
                raise RuntimeError(
                    f"parameter {name} occurs in optimizer groups {observed[id(param)]} and {group_index}"
                )
            observed[id(param)] = group_index
            tied_embedding = is_tied_embedding_parameter_name(name)
            tied_embedding_no_decay = tied_embedding and args.optimizer in {
                "muon",
                "tiller_v1",
            }
            required_wd = (
                0.0
                if is_no_decay_parameter(name, original) or tied_embedding_no_decay
                else float(args.weight_decay)
            )
            if weight_decay != required_wd:
                raise RuntimeError(
                    f"weight decay mismatch for {name}: {weight_decay} != {required_wd}"
                )
            names.append(name)
            element_count += int(param.numel())
        group_records.append(
            {
                "group_index": group_index,
                "lr_scale": 1.0,
                "weight_decay": weight_decay,
                "parameter_tensor_count": len(names),
                "parameter_element_count": element_count,
                "parameter_name_sha256": hashlib.sha256(
                    "\n".join(sorted(names)).encode("utf-8")
                ).hexdigest(),
            }
        )
    missing = sorted(name for param_id, (name, _) in named_parameters.items() if param_id not in observed)
    if missing:
        raise RuntimeError(f"trainable parameters missing from optimizer: {missing[:8]}")

    internal_scalars = {}
    from training.baseline_optimizers import (
        AdEMAMix,
        CAMEStyleAdamW,
        Lion,
        SOAPStyleAdamW,
        ScheduleFreeAdamW,
    )

    standard_types = (
        torch.optim.AdamW,
        torch.optim.Muon,
        CompositeOptimizer,
        Lion,
        AdEMAMix,
        ScheduleFreeAdamW,
        CAMEStyleAdamW,
        SOAPStyleAdamW,
    )
    for object_index, current in enumerate(_optimizer_tree(optimizer)):
        for attribute, value in vars(current).items():
            if attribute.endswith(("_lr_scale", "_lr_scale_final", "_weight_decay_scale")):
                _unit_scalar(value, label=f"{type(current).__name__}.{attribute}")
                internal_scalars[f"{object_index}:{type(current).__name__}.{attribute}"] = 1.0
        if isinstance(current, standard_types):
            continue
        provider = getattr(current, "lr_wd_fairness_audit", None)
        if provider is None:
            raise RuntimeError(
                f"nonstandard optimizer {type(current).__name__} does not expose "
                "lr_wd_fairness_audit(); campaign execution is refused"
            )
        declared = provider()
        if not isinstance(declared, dict) or not declared:
            raise RuntimeError(
                f"{type(current).__name__}.lr_wd_fairness_audit() must return a nonempty dict"
            )
        for label, value in declared.items():
            _unit_scalar(value, label=f"{type(current).__name__}.{label}")
            internal_scalars[f"{object_index}:{type(current).__name__}.{label}"] = 1.0
    return {
        "contract": EXACT_LR_WD_CONTRACT,
        "passed": True,
        "base_lr": float(args.lr),
        "minimum_lr": float(args.min_lr),
        "base_weight_decay": float(args.weight_decay),
        "group_count": len(group_records),
        "trainable_parameter_elements": sum(
            int(param.numel()) for _, param in named_parameters.values()
        ),
        "covered_parameter_elements": sum(
            record["parameter_element_count"] for record in group_records
        ),
        "groups": group_records,
        "internal_lr_wd_scalars": internal_scalars,
    }


def assert_optimizer_realized_lr(optimizer, scheduled_lr, args):
    if args.fairness_contract != EXACT_LR_WD_CONTRACT:
        return
    for group_index, group in enumerate(optimizer.param_groups):
        _unit_scalar(group.get("lr_scale", 1.0), label=f"param_groups[{group_index}].lr_scale")
        observed = float(group["lr"])
        if observed != float(scheduled_lr):
            raise RuntimeError(
                f"realized LR mismatch in param group {group_index}: "
                f"{observed} != {scheduled_lr}"
            )


def is_no_decay_parameter(name, param):
    return param.dim() < 2 or ".rlb_activation." in name


def is_tied_embedding_parameter_name(name):
    """Recognize tied embeddings before and after DDP adds ``module.``."""
    bare_name = name.removeprefix("module.")
    return bare_name in {"token_embedding.weight", "lm_head.weight"}


def split_decay_parameters(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_no_decay_parameter(name, param):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def is_rational_optimizer_parameter_name(name):
    return (
        ".rlb_activation." in name
        or name.endswith(".numerator")
        or name.endswith(".denominator")
    )


def rational_optimizer_layer_index(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def count_rational_optimizer_parameters(model):
    return sum(
        param.numel()
        for name, param in model.named_parameters()
        if param.requires_grad and is_rational_optimizer_parameter_name(name)
    )


def collect_rlb_optimizer_groups(model, args):
    groups = []
    for module_name, module in model.named_modules():
        activation = getattr(module, "rlb_activation", None)
        in_proj = getattr(module, "in_proj", None)
        out_proj = getattr(module, "out_proj", None)
        if activation is None or in_proj is None or out_proj is None:
            continue
        required = ("groups", "hidden_dim", "numerator", "denominator")
        if not isinstance(in_proj, nn.Linear) or not isinstance(out_proj, nn.Linear):
            continue
        if not all(hasattr(activation, attr) for attr in required):
            continue
        if not isinstance(activation.numerator, nn.Parameter):
            continue
        if not isinstance(activation.denominator, nn.Parameter):
            continue
        coeff_logits = getattr(activation, "coeff_logits", None)
        if coeff_logits is not None and not isinstance(coeff_logits, nn.Parameter):
            continue
        if coeff_logits is not None and coeff_logits.numel() == 0:
            coeff_logits = None
        hidden_dim = int(activation.hidden_dim)
        groups_count = int(activation.groups)
        if in_proj.weight.shape[0] != hidden_dim or out_proj.weight.shape[1] != hidden_dim:
            continue
        groups.append(
            {
                "mlp": module,
                "module": activation,
                "in_weight": in_proj.weight,
                "out_weight": out_proj.weight,
                "numerator": activation.numerator,
                "denominator": activation.denominator,
                "coeff_logits": coeff_logits,
                "centers": getattr(activation, "centers", None),
                "beta": getattr(activation, "beta", None),
                "coeff_limit": float(getattr(activation, "coeff_limit", 0.0)),
                "groups": groups_count,
                "hidden_dim": hidden_dim,
                "eps": float(getattr(activation, "eps", 0.0)),
                "layer_index": rational_optimizer_layer_index(module_name + "."),
                "num_layers": args.layers,
            }
        )
    return groups


def resolve_ademamix_warmup(value, steps):
    if value is None:
        return None
    if int(value) < 0:
        return max(1, int(round(0.15 * int(steps))))
    if int(value) == 0:
        return None
    return int(value)


def configure_optimizer(model, args):
    if args.optimizer not in ACTIVE_OPTIMIZERS:
        allowed = ", ".join(ACTIVE_OPTIMIZERS)
        raise ValueError(f"Accepted optimizer choices: {allowed}")
    decay, no_decay = split_decay_parameters(model)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
    if args.optimizer == "lion":
        from training.baseline_optimizers import Lion

        return Lion(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "ademamix":
        from training.baseline_optimizers import AdEMAMix

        beta3_warmup = resolve_ademamix_warmup(args.ademamix_beta3_warmup_steps, args.steps)
        alpha_warmup = resolve_ademamix_warmup(args.ademamix_alpha_warmup_steps, args.steps)
        return AdEMAMix(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.ademamix_beta3),
            eps=args.eps,
            weight_decay=args.weight_decay,
            alpha=args.ademamix_alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
        )
    if args.optimizer == "schedule_free_adamw":
        from training.baseline_optimizers import ScheduleFreeAdamW

        return ScheduleFreeAdamW(
            groups,
            lr=args.lr,
            beta1=args.schedule_free_beta1,
            beta2=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
            warmup_steps=args.schedule_free_warmup_steps,
        )
    if args.optimizer == "adafactor_came":
        from training.baseline_optimizers import CAMEStyleAdamW

        return CAMEStyleAdamW(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.came_beta3),
            eps=args.eps,
            weight_decay=args.weight_decay,
            factored_min_dim=args.factored_min_dim,
            clip_threshold=args.factored_clip_threshold,
            confidence_scale=args.came_confidence_scale,
            confidence_min=args.came_confidence_min,
            confidence_max=args.came_confidence_max,
        )
    if args.optimizer == "soap_adamw":
        from training.baseline_optimizers import SOAPStyleAdamW

        return SOAPStyleAdamW(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            precondition_frequency=args.soap_precondition_frequency,
            large_side_identity_threshold=args.soap_large_side_identity_threshold,
            one_sided=args.soap_one_sided,
        )
    if args.optimizer == "muon":
        muon_named = []
        adam_decay = []
        adam_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            no_decay_param = is_no_decay_parameter(name, param)
            tied_embedding = is_tied_embedding_parameter_name(name)
            if param.dim() == 2 and not no_decay_param and not tied_embedding:
                muon_named.append((name, param))
            elif no_decay_param or tied_embedding:
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)

        optimizers = []
        if muon_named:
            optimizers.append(
                torch.optim.Muon(
                    muon_named,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    momentum=args.muon_momentum,
                    ns_steps=args.muon_ns_steps,
                    adjust_lr_fn=args.muon_adjust_lr_fn,
                )
            )
        adam_groups = []
        if adam_decay:
            adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
        if adam_no_decay:
            adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
        if adam_groups:
            optimizers.append(
                torch.optim.AdamW(
                    adam_groups,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            )
        return CompositeOptimizer(optimizers)
    allowed = ", ".join(ACTIVE_OPTIMIZERS)
    raise ValueError(f"Accepted optimizer choices: {allowed}")

def learning_rate(step, args):
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return args.lr * float(step + 1) / float(args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + cosine * (args.lr - args.min_lr)


def write_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activation",
        choices=_ACCEPTED_ACTIVATIONS,
        default="silu",
        metavar="{silu,grain}",
    )
    parser.add_argument("--run-name", default="lm_optimizer_sweep")
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument(
        "--dataset-streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dataset-text-column", default="text")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--train-skip-documents", type=int, default=0)
    parser.add_argument("--validation-skip-documents", type=int, default=0)
    parser.add_argument("--train-skip-tokens", type=int, default=0)
    parser.add_argument("--validation-skip-tokens", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--hf-cache", default="experiments/cache/huggingface")
    parser.add_argument("--cache-dir", default="experiments/cache/tokens")
    parser.add_argument("--output-dir", default="experiments/runs/lm_optimizer_sweep")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-train-tokens", type=int, default=100_000_000)
    parser.add_argument("--max-val-tokens", type=int, default=2_000_000)
    parser.add_argument("--tokenize-batch-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--rational-init", choices=["silu"], default="silu")
    parser.add_argument("--post-rational-init", default="identity")
    parser.add_argument("--rational-group-size", type=int, default=256)
    parser.add_argument("--rational-max-groups", type=int, default=32)
    parser.add_argument("--rational-basis-eps", type=float, default=1e-6)
    parser.add_argument("--optimizer", choices=ACTIVE_OPTIMIZERS, default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--fairness-contract",
        choices=["none", EXACT_LR_WD_CONTRACT],
        default="none",
        help="Verify the configured LR schedule and weight-decay routing.",
    )
    parser.add_argument(
        "--experiment-identity",
        choices=["none", TILLER_EXPERIMENT_IDENTITY],
        default="none",
        help="Validate a registered matched-experiment identity.",
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)

    # Published optimizer-baseline controls.
    parser.add_argument("--factored-min-dim", type=int, default=128)
    parser.add_argument("--factored-clip-threshold", type=float, default=1.0)
    parser.add_argument("--ademamix-beta3", type=float, default=0.9999)
    parser.add_argument("--ademamix-alpha", type=float, default=5.0)
    parser.add_argument("--ademamix-beta3-warmup-steps", type=int, default=-1)
    parser.add_argument("--ademamix-alpha-warmup-steps", type=int, default=-1)
    parser.add_argument("--schedule-free-beta1", type=float, default=0.9)
    parser.add_argument("--schedule-free-warmup-steps", type=int, default=0)
    parser.add_argument("--came-beta3", type=float, default=0.999)
    parser.add_argument("--came-confidence-scale", type=float, default=1.0)
    parser.add_argument("--came-confidence-min", type=float, default=0.25)
    parser.add_argument("--came-confidence-max", type=float, default=4.0)
    parser.add_argument("--soap-precondition-frequency", type=int, default=50)
    parser.add_argument(
        "--soap-large-side-identity-threshold", type=int, default=2048
    )
    parser.add_argument(
        "--soap-one-sided",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument(
        "--muon-adjust-lr-fn",
        choices=["original", "match_rms_adamw"],
        default="match_rms_adamw",
    )

    # These two flags are retained because every published row records SAM off.
    parser.add_argument("--sam-rho", type=float, default=0.0)
    parser.add_argument(
        "--sam-adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--early-stop-min-step", type=int, default=0)
    parser.add_argument("--early-stop-max-val-loss", type=float, default=0.0)
    parser.add_argument("--early-stop-loss-increase", type=float, default=0.0)
    parser.add_argument(
        "--timing-guard-max-seconds-per-step", type=float, default=0.0
    )
    parser.add_argument(
        "--timing-guard-max-optimizer-step-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument("--timing-guard-min-step", type=int, default=0)
    parser.add_argument("--probe-batch-size", type=int, default=2)
    parser.add_argument(
        "--telemetry-grain-stat-every",
        dest="telemetry_rlb_stat_every",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--telemetry-rlb-stat-every",
        dest="telemetry_rlb_stat_every",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--telemetry-grain-stat-samples",
        dest="telemetry_rlb_stat_samples",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--telemetry-rlb-stat-samples",
        dest="telemetry_rlb_stat_samples",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--telemetry-denominator-probe-points", type=int, default=129
    )
    parser.add_argument("--matrix-spectrum-interval", type=int, default=500)
    parser.add_argument("--matrix-spectrum-max-dim", type=int, default=512)
    parser.add_argument(
        "--grain-init-gauge-log-scale",
        dest="rlb_init_gauge_log_scale",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--rlb-init-gauge-log-scale",
        dest="rlb_init_gauge_log_scale",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--grain-init-gauge-seed",
        dest="rlb_init_gauge_seed",
        type=int,
        default=424242,
    )
    parser.add_argument(
        "--rlb-init-gauge-seed",
        dest="rlb_init_gauge_seed",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--resume-checkpoint-interval", type=int, default=0)
    return parser.parse_args()


def validate_optimizer_protocol(args):
    if args.optimizer not in ACTIVE_OPTIMIZERS:
        allowed = ", ".join(ACTIVE_OPTIMIZERS)
        raise ValueError(f"Accepted optimizer choices: {allowed}")
    if (
        args.optimizer in RATIONAL_SPECIFIC_OPTIMIZERS
        and args.activation not in GRAIN_ACTIVATION_IDS
    ):
        raise ValueError(
            f"optimizer {args.optimizer} requires the GRAIN activation"
        )
    if float(args.sam_rho) != 0.0 or bool(args.sam_adaptive):
        raise ValueError("the published experiment protocol requires SAM to remain disabled")
    if args.fairness_contract == EXACT_LR_WD_CONTRACT:
        if not 0.0 <= float(args.min_lr) <= float(args.lr):
            raise ValueError("the exact LR/WD contract requires 0 <= min_lr <= lr")
        if float(args.weight_decay) < 0.0:
            raise ValueError("the exact LR/WD contract requires nonnegative weight decay")
        if args.resume_checkpoint is not None:
            raise ValueError("fairness-contracted runs may not resume from an external checkpoint")
    if (
        args.experiment_identity == TILLER_EXPERIMENT_IDENTITY
        and args.fairness_contract != EXACT_LR_WD_CONTRACT
    ):
        raise ValueError("the TILLER matrix identity requires the exact LR/WD contract")


def _resume_rank_path(path: Path, rank: int) -> Path:
    return Path(f"{path}.rank{rank}.pt")


def _resume_contract(args, world_size: int) -> dict:
    return {
        "schema": "transformer_lm_exact_resume_v1",
        "activation": args.activation,
        "optimizer": args.optimizer,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "world_size": int(world_size),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "seq_len": int(args.seq_len),
        "lr": float(args.lr),
        "min_lr": float(args.min_lr),
        "warmup_steps": int(args.warmup_steps),
        "weight_decay": float(args.weight_decay),
        "beta1": float(args.beta1),
        "beta2": float(args.beta2),
        "eps": float(args.eps),
        "grad_clip": float(args.grad_clip),
    }


def _cpu_probe_resume_state(probe_state):
    if probe_state is None:
        return None
    return {
        key: (
            None
            if probe_state.get(key) is None
            else probe_state[key].detach().cpu()
        )
        for key in ("first_logits", "prev_logits")
    }


def _save_exact_resume_checkpoint(
    path: Path,
    *,
    args,
    rank: int,
    world_size: int,
    is_distributed: bool,
    device: torch.device,
    model,
    optimizer,
    train_generator: torch.Generator,
    probe_state,
    completed_step: int,
    best_val_loss: float,
    step_times: list[float],
    optimizer_step_times: list[float],
    grad_clip_observed_steps: int,
    grad_clip_triggered_steps: int,
    run_cuda_max_memory_allocated: int,
    run_cuda_max_memory_reserved: int,
    active_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rank_path = _resume_rank_path(path, rank)
    rank_tmp = Path(f"{rank_path}.tmp")
    if rank_tmp.exists():
        rank_tmp.unlink()
    torch.save(
        {
            "schema": "transformer_lm_exact_resume_rank_v1",
            "rank": int(rank),
            "completed_step": int(completed_step),
            "train_generator_state": train_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device),
            "probe_state": _cpu_probe_resume_state(probe_state),
        },
        rank_tmp,
    )

    main_tmp = Path(f"{path}.tmp")
    if rank == 0:
        if main_tmp.exists():
            main_tmp.unlink()
        raw_model = (
            model.module
            if isinstance(model, nn.parallel.DistributedDataParallel)
            else model
        )
        torch.save(
            {
                "schema": "transformer_lm_exact_resume_v1",
                "contract": _resume_contract(args, world_size),
                "completed_step": int(completed_step),
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": float(best_val_loss),
                "step_times": list(step_times),
                "optimizer_step_times": list(optimizer_step_times),
                "grad_clip_observed_steps": int(
                    grad_clip_observed_steps
                ),
                "grad_clip_triggered_steps": int(
                    grad_clip_triggered_steps
                ),
                "run_cuda_max_memory_allocated": int(
                    run_cuda_max_memory_allocated
                ),
                "run_cuda_max_memory_reserved": int(
                    run_cuda_max_memory_reserved
                ),
                "active_seconds": float(active_seconds),
            },
            main_tmp,
        )

    if is_distributed:
        dist.barrier()
    rank_tmp.replace(rank_path)
    if is_distributed:
        dist.barrier()
    if rank == 0:
        main_tmp.replace(path)
    if is_distributed:
        dist.barrier()


def _load_exact_resume_checkpoint(
    path: Path,
    *,
    args,
    rank: int,
    world_size: int,
    is_distributed: bool,
    device: torch.device,
    model,
    optimizer,
    train_generator: torch.Generator,
    probe_state,
) -> dict | None:
    if not path.is_file():
        return None
    rank_path = _resume_rank_path(path, rank)
    if not rank_path.is_file():
        raise RuntimeError(
            f"missing exact-resume rank state: {rank_path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rank_payload = torch.load(
        rank_path, map_location="cpu", weights_only=False
    )
    if (
        payload.get("schema") != "transformer_lm_exact_resume_v1"
        or payload.get("contract") != _resume_contract(args, world_size)
        or rank_payload.get("schema")
        != "transformer_lm_exact_resume_rank_v1"
        or rank_payload.get("rank") != rank
        or rank_payload.get("completed_step")
        != payload.get("completed_step")
    ):
        raise RuntimeError("exact-resume checkpoint contract differs")

    raw_model = (
        model.module
        if isinstance(model, nn.parallel.DistributedDataParallel)
        else model
    )
    raw_model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    train_generator.set_state(rank_payload["train_generator_state"])
    torch.set_rng_state(rank_payload["torch_rng_state"])
    torch.cuda.set_rng_state(rank_payload["cuda_rng_state"], device)
    stored_probe = rank_payload.get("probe_state")
    if probe_state is not None and stored_probe is not None:
        for key in ("first_logits", "prev_logits"):
            value = stored_probe.get(key)
            probe_state[key] = (
                None if value is None else value.to(device=device)
            )
    if is_distributed:
        dist.barrier()
    return payload


def main():
    args = parse_args()
    validate_optimizer_protocol(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.prepare_only:
        train_tokens = load_or_tokenize(args, "train", args.max_train_tokens)
        val_tokens = load_or_tokenize(args, "validation", args.max_val_tokens)
        print(
            json.dumps({"event": "prepared", "train_tokens": train_tokens.numel(), "val_tokens": val_tokens.numel()}),
            flush=True,
        )
        os._exit(0)

    is_distributed, rank, local_rank, world_size, device = setup_distributed()
    if device.type != "cuda":
        raise RuntimeError("this benchmark is intended to run on CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_tokens = load_or_tokenize(args, "train", args.max_train_tokens)
    val_tokens = load_or_tokenize(args, "validation", args.max_val_tokens)
    train_tokens = train_tokens.pin_memory()
    val_tokens = val_tokens.pin_memory()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer,
        cache_dir=args.hf_cache,
        revision=args.tokenizer_revision,
    )
    vocab_size = len(tokenizer)
    global_tokens = args.batch_size * args.grad_accum * world_size * args.seq_len
    if args.steps <= 0:
        args.steps = max(1, train_tokens.numel() // global_tokens)
    model = CausalTransformer(args, vocab_size)
    initial_state_sha256 = model_state_sha256(model) if rank == 0 else None
    model = model.to(device)
    apply_rlb_positive_gauge(
        model, args.rlb_init_gauge_log_scale, args.rlb_init_gauge_seed
    )
    enable_rlb_training_telemetry(model, args)
    param_count = sum(param.numel() for param in model.parameters())
    tiller_experiment_identity = audit_tiller_experiment_identity(
        args,
        world_size=world_size,
        global_tokens=global_tokens,
        train_token_count=train_tokens.numel(),
        val_token_count=val_tokens.numel(),
        parameter_count=param_count,
    )
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    rational_optimizer_parameter_count = count_rational_optimizer_parameters(model)
    optimizer = configure_optimizer(model, args)
    optimizer_lr_wd_fairness = audit_optimizer_lr_wd_fairness(
        model, optimizer, args
    )
    offsets = torch.arange(args.seq_len + 1)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(args.seed + 997 * rank)
    out_path = Path(args.output_dir) / args.run_name / f"{args.activation}.jsonl"
    probe_state = prepare_probe_batch(val_tokens, args, offsets, rank, device, out_path)
    resume_checkpoint = (
        None
        if args.resume_checkpoint is None
        else Path(args.resume_checkpoint)
    )
    resume_payload = (
        None
        if resume_checkpoint is None
        else _load_exact_resume_checkpoint(
            resume_checkpoint,
            args=args,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
            device=device,
            model=model,
            optimizer=optimizer,
            train_generator=train_generator,
            probe_state=probe_state,
        )
    )
    resume_step = (
        0
        if resume_payload is None
        else int(resume_payload["completed_step"])
    )
    if resume_step < 0 or resume_step >= args.steps:
        raise RuntimeError(
            f"invalid exact-resume completed step: {resume_step}"
        )
    grain_cfg = (
        grain_settings(
            args.ffn_dim,
            args.rational_group_size,
            args.rational_max_groups,
        )
        if args.activation in GRAIN_ACTIVATION_IDS
        else None
    )

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)
    slurm_node = os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST")
    timing_attempt_id = f"{slurm_job_id or 'manual'}:{slurm_restart_count}:{int(time.time())}"
    optimizer_telemetry_steps = parse_optimizer_telemetry_steps(
        os.environ.get("TILLER_OPTIMIZER_TELEMETRY_STEPS"),
        args.steps,
    )
    strict_gpu_metadata = os.environ.get("TILLER_STRICT_GPU_METADATA", "0") == "1"
    ddp_sync_check_interval = max(
        0,
        int(os.environ.get("RLB_DDP_SYNC_CHECK_INTERVAL", "0") or 0),
    )
    gpu_metadata = nvidia_smi_metadata(strict=strict_gpu_metadata) if rank == 0 else None
    train_token_sample_sha256 = tensor_sample_sha256(train_tokens) if rank == 0 else None
    val_token_sample_sha256 = tensor_sample_sha256(val_tokens) if rank == 0 else None
    first_batch_index_sha256 = (
        sampled_start_fingerprint(
            train_tokens.numel(),
            args.seq_len,
            args.batch_size,
            args.grad_accum,
            args.seed,
        )
        if rank == 0
        else None
    )
    validation_index_sha256 = (
        sampled_start_fingerprint(
            val_tokens.numel(),
            args.seq_len,
            args.batch_size,
            args.eval_batches,
            args.seed + 1_000_003,
        )
        if rank == 0
        else None
    )

    config_record = {
        "event": "config",
        "activation": args.activation,
        "activation_display_name": (
            "GRAIN"
            if args.activation in GRAIN_ACTIVATION_IDS
            else "SwiGLU"
        ),
        "slurm_job_id": slurm_job_id,
        "slurm_restart_count": slurm_restart_count,
        "slurm_node": slurm_node,
        "timing_attempt_id": timing_attempt_id,
        "optimizer_telemetry_steps": (
            None
            if optimizer_telemetry_steps is None
            else list(optimizer_telemetry_steps)
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "gpu_metadata": gpu_metadata,
        "initial_state_sha256": initial_state_sha256,
        "train_token_sample_sha256": train_token_sample_sha256,
        "val_token_sample_sha256": val_token_sample_sha256,
        "first_batch_index_sha256": first_batch_index_sha256,
        "validation_index_sha256": validation_index_sha256,
        "timing_guard_max_seconds_per_step": (
            args.timing_guard_max_seconds_per_step
        ),
        "timing_guard_max_optimizer_step_seconds": (
            args.timing_guard_max_optimizer_step_seconds
        ),
        "timing_guard_min_step": args.timing_guard_min_step,
        "resume_checkpoint": (
            None
            if resume_checkpoint is None
            else str(resume_checkpoint.resolve())
        ),
        "resume_checkpoint_interval": args.resume_checkpoint_interval,
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "dataset_streaming": args.dataset_streaming,
        "dataset_text_column": args.dataset_text_column,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "train_skip_documents": args.train_skip_documents,
        "validation_skip_documents": args.validation_skip_documents,
        "train_skip_tokens": args.train_skip_tokens,
        "validation_skip_tokens": args.validation_skip_tokens,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "train_tokens": train_tokens.numel(),
        "val_tokens": val_tokens.numel(),
        "world_size": world_size,
        "batch_size_per_gpu": args.batch_size,
        "global_tokens_per_step": global_tokens,
        "grad_accum": args.grad_accum,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "layers": args.layers,
        "d_model": args.d_model,
        "heads": args.heads,
        "ffn_dim": args.ffn_dim,
        "params": param_count,
        "init_std": args.init_std,
        "rational_init": args.rational_init,
        "post_rational_init": args.post_rational_init,
        "rational_group_size": args.rational_group_size,
        "rational_max_groups": args.rational_max_groups,
        "rational_basis_eps": args.rational_basis_eps,
        "grain_init": None if grain_cfg is None else grain_cfg["base_init"],
        "grain_centers": None if grain_cfg is None else grain_cfg["centers"],
        "grain_coefficient_limit": (
            None if grain_cfg is None else grain_cfg["coeff_limit"]
        ),
        "grain_groups": None if grain_cfg is None else grain_cfg["groups"],
        "grain_hidden_dim": (
            None if grain_cfg is None else grain_cfg["hidden_dim"]
        ),
        "grain_input_affine": (
            None if grain_cfg is None else grain_cfg["input_affine"]
        ),
        "grain_fused": None if grain_cfg is None else grain_cfg["fused"],
        "grain_center_odd": (
            None if grain_cfg is None else grain_cfg["center_odd"]
        ),
        "grain_train_centers": (
            None if grain_cfg is None else grain_cfg["train_centers"]
        ),
        "grain_atom_scale_init": (
            None if grain_cfg is None else grain_cfg["atom_scale_init"]
        ),
        "grain_atom_scale_limit": (
            None if grain_cfg is None else grain_cfg["atom_scale_limit"]
        ),
        "grain_live_stats_scope": rlb_live_stats_scope(model),
        "grain_ddp_sync_check_interval": ddp_sync_check_interval,
        "grain_telemetry_stat_every": args.telemetry_rlb_stat_every,
        "grain_telemetry_stat_samples": args.telemetry_rlb_stat_samples,
        "telemetry_denominator_probe_points": (
            args.telemetry_denominator_probe_points
        ),
        "matrix_spectrum_interval": args.matrix_spectrum_interval,
        "matrix_spectrum_max_dim": args.matrix_spectrum_max_dim,
        "optimizer": args.optimizer,
        "fairness_contract": args.fairness_contract,
        "experiment_identity": args.experiment_identity,
        "optimizer_lr_wd_fairness": optimizer_lr_wd_fairness,
        "tiller_experiment_identity": tiller_experiment_identity,
        "source_manifest_sha256": os.environ.get(
            "RATIONALOPT_SOURCE_MANIFEST_SHA256"
        ),
        "source_manifest_row_sha256": os.environ.get(
            "RATIONALOPT_SOURCE_MANIFEST_ROW_SHA256"
        ),
        "source_manifest_row_id": os.environ.get(
            "RATIONALOPT_SOURCE_MANIFEST_ROW_ID"
        ),
        "source_manifest_row_index": os.environ.get(
            "RATIONALOPT_SOURCE_MANIFEST_ROW_INDEX"
        ),
        "source_freeze_sha256": os.environ.get(
            "RATIONALOPT_SOURCE_FREEZE_SHA256"
        ),
        "optimizer_lr": args.lr,
        "optimizer_min_lr": args.min_lr,
        "optimizer_weight_decay": args.weight_decay,
        "optimizer_beta1": args.beta1,
        "optimizer_beta2": args.beta2,
        "optimizer_eps": args.eps,
        "warmup_steps": args.warmup_steps,
        "grad_clip": args.grad_clip,
        "log_interval": args.log_interval,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "early_stop_min_step": args.early_stop_min_step,
        "early_stop_max_val_loss": args.early_stop_max_val_loss,
        "early_stop_loss_increase": args.early_stop_loss_increase,
        "factored_min_dim": (
            args.factored_min_dim
            if args.optimizer == "adafactor_came"
            else None
        ),
        "factored_clip_threshold": (
            args.factored_clip_threshold
            if args.optimizer == "adafactor_came"
            else None
        ),
        "ademamix_beta3": (
            args.ademamix_beta3 if args.optimizer == "ademamix" else None
        ),
        "ademamix_alpha": (
            args.ademamix_alpha if args.optimizer == "ademamix" else None
        ),
        "ademamix_beta3_warmup_steps": (
            resolve_ademamix_warmup(
                args.ademamix_beta3_warmup_steps, args.steps
            )
            if args.optimizer == "ademamix"
            else None
        ),
        "ademamix_alpha_warmup_steps": (
            resolve_ademamix_warmup(
                args.ademamix_alpha_warmup_steps, args.steps
            )
            if args.optimizer == "ademamix"
            else None
        ),
        "schedule_free_beta1": (
            args.schedule_free_beta1
            if args.optimizer == "schedule_free_adamw"
            else None
        ),
        "schedule_free_warmup_steps": (
            args.schedule_free_warmup_steps
            if args.optimizer == "schedule_free_adamw"
            else None
        ),
        "came_beta3": (
            args.came_beta3 if args.optimizer == "adafactor_came" else None
        ),
        "came_confidence_scale": (
            args.came_confidence_scale
            if args.optimizer == "adafactor_came"
            else None
        ),
        "came_confidence_min": (
            args.came_confidence_min
            if args.optimizer == "adafactor_came"
            else None
        ),
        "came_confidence_max": (
            args.came_confidence_max
            if args.optimizer == "adafactor_came"
            else None
        ),
        "soap_precondition_frequency": (
            args.soap_precondition_frequency
            if args.optimizer == "soap_adamw"
            else None
        ),
        "soap_large_side_identity_threshold": (
            args.soap_large_side_identity_threshold
            if args.optimizer == "soap_adamw"
            else None
        ),
        "soap_one_sided": (
            args.soap_one_sided if args.optimizer == "soap_adamw" else None
        ),
        "muon_adjust_lr_fn": (
            args.muon_adjust_lr_fn if args.optimizer == "muon" else None
        ),
        "muon_momentum": (
            args.muon_momentum if args.optimizer == "muon" else None
        ),
        "muon_ns_steps": (
            args.muon_ns_steps if args.optimizer == "muon" else None
        ),
        "sam_rho": args.sam_rho,
        "sam_adaptive": args.sam_adaptive,
        "grain_coefficient_parameters": (
            rational_optimizer_parameter_count
            if args.optimizer in RATIONAL_SPECIFIC_OPTIMIZERS
            else None
        ),
    }
    for env_name in (
        "TILLER_ROW_ID",
        "TILLER_ARM_ID",
        "TILLER_DESIGN_VERSION",
        "TILLER_MANIFEST_SHA256",
        "TILLER_FREEZE_SHA256",
        "TILLER_RUNTIME_FREEZE_SHA256",
        "TILLER_ENVIRONMENT_SHA256",
        "TILLER_ARRAY_JOB_ID",
        "TILLER_ARRAY_TASK_ID",
        "TILLER_ARRAY_TASK_COUNT",
        "TILLER_ARRAY_TASK_MIN",
        "TILLER_ARRAY_TASK_MAX",
        "TILLER_ARRAY_TASK_STEP",
        "TILLER_ARRAY_TASK_THROTTLE",
        "TILLER_SPOOLED_LAUNCHER_SHA256",
        "TILLER_RUNTIME_ENVIRONMENT_VALIDATED",
        "TILLER_DISTRIBUTION_CLOSURE_SHA256",
        "TILLER_SUBMISSION_NONCE",
        "TILLER_SUBMISSION_INTENT_SHA256",
        "TILLER_SUBMISSION_RESULT_SHA256",
        "TILLER_SUBMISSION_DEPENDENCY_JOB_ID",
        "TILLER_PREFLIGHT_ATTEMPT_ID",
        "TILLER_PREFLIGHT_SPOOLED_LAUNCHER_SHA256",
    ):
        config_record[env_name.lower()] = os.environ.get(env_name)
    rank0_print(rank, json.dumps(config_record, sort_keys=True))
    if rank == 0 and resume_payload is None:
        write_jsonl(out_path, config_record)
    elif rank == 0:
        write_jsonl(
            out_path,
            {
                "event": "resumed",
                "activation": args.activation,
                "step": resume_step,
                "checkpoint": str(resume_checkpoint.resolve()),
                "slurm_job_id": slurm_job_id,
                "slurm_restart_count": slurm_restart_count,
                "slurm_node": slurm_node,
                "timing_attempt_id": timing_attempt_id,
            },
        )

    step_times = (
        []
        if resume_payload is None
        else [float(value) for value in resume_payload["step_times"]]
    )
    optimizer_step_times = (
        []
        if resume_payload is None
        else [
            float(value)
            for value in resume_payload.get("optimizer_step_times", [])
        ]
    )
    optimizer_start_event = (
        torch.cuda.Event(enable_timing=True)
        if device.type == "cuda"
        else None
    )
    optimizer_end_event = (
        torch.cuda.Event(enable_timing=True)
        if device.type == "cuda"
        else None
    )
    loss_since_log = 0.0
    steps_since_log = 0
    best_val_loss = (
        math.inf
        if resume_payload is None
        else float(resume_payload["best_val_loss"])
    )
    stop_reason = None
    stop_step = None
    grad_clip_observed_steps = (
        0
        if resume_payload is None
        else int(resume_payload["grad_clip_observed_steps"])
    )
    grad_clip_triggered_steps = (
        0
        if resume_payload is None
        else int(resume_payload["grad_clip_triggered_steps"])
    )
    run_cuda_max_memory_allocated = (
        0
        if resume_payload is None
        else int(resume_payload["run_cuda_max_memory_allocated"])
    )
    run_cuda_max_memory_reserved = (
        0
        if resume_payload is None
        else int(resume_payload["run_cuda_max_memory_reserved"])
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    prior_active_seconds = (
        0.0
        if resume_payload is None
        else float(resume_payload["active_seconds"])
    )
    start_time = time.perf_counter() - prior_active_seconds
    realized_lr_trace = hashlib.sha256()
    realized_lr_audit_steps = 0
    for step in range(resume_step, args.steps):
        model.train()
        step_start = time.perf_counter()
        will_log = (step + 1) % args.log_interval == 0 or step == 0 or step + 1 == args.steps
        will_eval = args.eval_interval > 0 and (
            step == 0 or (step + 1) % args.eval_interval == 0 or step + 1 == args.steps
        )
        capture_step_telemetry = will_log and rank == 0
        if capture_step_telemetry and device.type == "cuda":
            run_cuda_max_memory_allocated = max(
                run_cuda_max_memory_allocated,
                int(torch.cuda.max_memory_allocated(device)),
            )
            run_cuda_max_memory_reserved = max(
                run_cuda_max_memory_reserved,
                int(torch.cuda.max_memory_reserved(device)),
            )
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        capture_optimizer_telemetry = capture_step_telemetry and (
            optimizer_telemetry_steps is None
            or step + 1 in optimizer_telemetry_steps
        )
        set_optimizer_telemetry_capture(optimizer, capture_optimizer_telemetry)
        forward_backward_start = time.perf_counter()
        forward_backward_seconds = None
        optimizer_step_seconds = None
        grad_global_norm_before_clip = None
        grad_clip_triggered = False

        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        assert_optimizer_realized_lr(optimizer, lr, args)
        optimizer.zero_grad(set_to_none=True)

        local_loss = 0.0
        rho = 0.0
        for micro_step in range(args.grad_accum):
            x, y = sample_batch(
                train_tokens,
                args.batch_size,
                args.seq_len,
                offsets,
                train_generator,
                device,
            )
            sync_context = (
                model.no_sync()
                if is_distributed and micro_step + 1 < args.grad_accum
                else contextlib.nullcontext()
            )
            with sync_context:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.reshape(-1)
                )
                (loss / args.grad_accum).backward()
            local_loss += float(loss.item())
        if capture_step_telemetry and device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_backward_seconds = (
            time.perf_counter() - forward_backward_start
        )
        grad_global_norm_before_clip, grad_clip_triggered = (
            clip_or_measure_gradients(
                model,
                args.grad_clip,
                capture_step_telemetry,
            )
        )
        optimizer_step_start = time.perf_counter()
        if optimizer_start_event is not None:
            optimizer_start_event.record()
        optimizer.step()
        if optimizer_end_event is not None:
            optimizer_end_event.record()
        assert_optimizer_realized_lr(optimizer, lr, args)
        if args.fairness_contract == EXACT_LR_WD_CONTRACT:
            realized_lr_trace.update(f"{step + 1}:{float(lr).hex()}\n".encode("ascii"))
            realized_lr_audit_steps += 1
        ddp_parameter_sync_max_abs = None
        grain_coefficient_sync_max_abs = None
        if (
            args.optimizer in RLB_MATRIX_SYNC_OPTIMIZERS
            and ddp_sync_check_interval > 0
            and (
                step == 0
                or (step + 1) % ddp_sync_check_interval == 0
                or will_eval
                or step + 1 == args.steps
            )
        ):
            ddp_parameter_sync_max_abs = assert_rlb_matrix_parameters_synced(
                model,
                args,
                device,
                is_distributed,
            )
        if args.optimizer in RLB_COEFFICIENT_SYNC_OPTIMIZERS and (
            step == 0 or will_eval or step + 1 == args.steps
        ):
            grain_coefficient_sync_max_abs = assert_rlb_coefficient_parameters_synced(
                model,
                args,
                device,
                is_distributed,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            if optimizer_start_event is None or optimizer_end_event is None:
                raise RuntimeError("CUDA optimizer timing events were lost")
            optimizer_step_seconds = (
                optimizer_start_event.elapsed_time(optimizer_end_event)
                / 1000.0
            )
            run_cuda_max_memory_allocated = max(
                run_cuda_max_memory_allocated,
                int(torch.cuda.max_memory_allocated(device)),
            )
            run_cuda_max_memory_reserved = max(
                run_cuda_max_memory_reserved,
                int(torch.cuda.max_memory_reserved(device)),
            )
        else:
            optimizer_step_seconds = (
                time.perf_counter() - optimizer_step_start
            )

        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        optimizer_step_times.append(optimizer_step_seconds)
        mean_loss = reduce_mean(local_loss / args.grad_accum, device, is_distributed)
        loss_since_log += mean_loss
        steps_since_log += 1
        grad_clip_observed_steps += 1
        if grad_clip_triggered:
            grad_clip_triggered_steps += 1
        if not math.isfinite(mean_loss):
            stop_reason = "nonfinite_train_loss"
            stop_step = step + 1
            stop_record = {
                "event": "stopped_early",
                "activation": args.activation,
                "step": stop_step,
                "reason": stop_reason,
                "loss": mean_loss,
                "timing_attempt_id": timing_attempt_id,
                "active_seconds_after_event": time.perf_counter() - start_time,
            }
            rank0_print(rank, json.dumps(stop_record, sort_keys=True))
            if rank == 0:
                write_jsonl(out_path, stop_record)
            break

        if will_log:
            recent = step_times[-args.log_interval :]
            recent_optimizer = optimizer_step_times[
                -args.log_interval :
            ]
            mean_recent_step = sum(recent) / len(recent)
            mean_recent_optimizer = (
                sum(recent_optimizer) / len(recent_optimizer)
            )
            tokens_per_second = global_tokens / mean_recent_step
            record = {
                "event": "train",
                "activation": args.activation,
                "step": step + 1,
                "loss": loss_since_log / max(1, steps_since_log),
                "lr": lr,
                "sam_rho": rho,
                "tokens_per_second": tokens_per_second,
                "seconds_per_step": mean_recent_step,
                "grad_global_norm_before_clip": _finite_float(grad_global_norm_before_clip),
                "grad_clip_triggered": bool(grad_clip_triggered),
                "grad_clip_threshold": args.grad_clip,
                "forward_backward_seconds": _finite_float(forward_backward_seconds),
                "optimizer_step_seconds": _finite_float(
                    mean_recent_optimizer
                ),
                "ddp_parameter_sync_max_abs": _finite_float(ddp_parameter_sync_max_abs),
                "grain_coefficient_sync_max_abs": _finite_float(grain_coefficient_sync_max_abs),
                "timing_attempt_id": timing_attempt_id,
                "active_seconds_after_event": time.perf_counter() - start_time,
            }
            if device.type == "cuda":
                record["cuda_max_memory_allocated"] = int(torch.cuda.max_memory_allocated(device))
                record["cuda_max_memory_reserved"] = int(torch.cuda.max_memory_reserved(device))
            if rank == 0:
                record.update(collect_optimizer_telemetry(optimizer))
                record.update(collect_rlb_telemetry(model, args))
            record["active_seconds_after_event"] = time.perf_counter() - start_time
            loss_since_log = 0.0
            steps_since_log = 0
            rank0_print(rank, json.dumps(record, sort_keys=True))
            if rank == 0:
                write_jsonl(out_path, record)

            guard_limit = float(args.timing_guard_max_seconds_per_step)
            guard_min_step = max(1, int(args.timing_guard_min_step))
            if guard_limit > 0.0 and step + 1 >= guard_min_step:
                guard_step_time = reduce_max(mean_recent_step, device, is_distributed)
                if guard_step_time > guard_limit:
                    guard_record = {
                        "event": "timing_guard_failed",
                        "activation": args.activation,
                        "step": step + 1,
                        "seconds_per_step": guard_step_time,
                        "max_seconds_per_step": guard_limit,
                        "timing_guard_min_step": guard_min_step,
                        "slurm_job_id": slurm_job_id,
                        "slurm_restart_count": slurm_restart_count,
                        "slurm_node": slurm_node,
                        "timing_attempt_id": timing_attempt_id,
                        "active_seconds_after_event": time.perf_counter() - start_time,
                    }
                    rank0_print(rank, json.dumps(guard_record, sort_keys=True))
                    if rank == 0:
                        write_jsonl(out_path, guard_record)
                    if is_distributed:
                        dist.barrier()
                    cleanup_distributed(is_distributed)
                    raise SystemExit(88)

            optimizer_guard_limit = float(
                args.timing_guard_max_optimizer_step_seconds
            )
            if (
                optimizer_guard_limit > 0.0
                and step + 1 >= guard_min_step
            ):
                guard_optimizer_time = reduce_max(
                    mean_recent_optimizer,
                    device,
                    is_distributed,
                )
                if guard_optimizer_time >= optimizer_guard_limit:
                    guard_record = {
                        "event": "optimizer_timing_guard_failed",
                        "activation": args.activation,
                        "step": step + 1,
                        "optimizer_step_seconds": guard_optimizer_time,
                        "max_optimizer_step_seconds": (
                            optimizer_guard_limit
                        ),
                        "timing_guard_min_step": guard_min_step,
                        "slurm_job_id": slurm_job_id,
                        "slurm_restart_count": slurm_restart_count,
                        "slurm_node": slurm_node,
                        "timing_attempt_id": timing_attempt_id,
                        "active_seconds_after_event": (
                            time.perf_counter() - start_time
                        ),
                    }
                    rank0_print(
                        rank,
                        json.dumps(guard_record, sort_keys=True),
                    )
                    if rank == 0:
                        write_jsonl(out_path, guard_record)
                    if is_distributed:
                        dist.barrier()
                    cleanup_distributed(is_distributed)
                    raise SystemExit(88)

        if will_eval:
            eval_start = time.perf_counter()
            val_loss = evaluate(model, val_tokens, args, offsets, rank, world_size, device, is_distributed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            active_seconds_at_val_loss = time.perf_counter() - start_time
            record = {
                "event": "eval",
                "activation": args.activation,
                "step": step + 1,
                "val_loss": val_loss,
                "val_ppl": math.exp(min(20.0, val_loss)),
                "grain_coefficient_sync_max_abs": _finite_float(grain_coefficient_sync_max_abs),
                "timing_attempt_id": timing_attempt_id,
                "active_seconds_at_val_loss": active_seconds_at_val_loss,
            }
            record.update(evaluate_probe(model, probe_state, device, is_distributed))
            if rank == 0 and args.matrix_spectrum_interval > 0 and (
                step == 0 or (step + 1) % args.matrix_spectrum_interval == 0 or step + 1 == args.steps
            ):
                record.update(collect_matrix_spectrum_telemetry(model, args))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                run_cuda_max_memory_allocated = max(
                    run_cuda_max_memory_allocated,
                    int(torch.cuda.max_memory_allocated(device)),
                )
                run_cuda_max_memory_reserved = max(
                    run_cuda_max_memory_reserved,
                    int(torch.cuda.max_memory_reserved(device)),
                )
            record["active_seconds_after_event"] = time.perf_counter() - start_time
            record["eval_seconds"] = record["active_seconds_after_event"] - (
                eval_start - start_time
            )
            rank0_print(rank, json.dumps(record, sort_keys=True))
            if rank == 0:
                write_jsonl(out_path, record)

            current_step = step + 1
            previous_best = best_val_loss
            nonfinite_val = not math.isfinite(val_loss)
            if not nonfinite_val:
                best_val_loss = min(best_val_loss, val_loss)
            min_step_met = current_step >= max(1, int(args.early_stop_min_step))
            max_loss = float(args.early_stop_max_val_loss)
            loss_increase = float(args.early_stop_loss_increase)
            too_large = max_loss > 0.0 and val_loss > max_loss
            worsened = (
                loss_increase > 0.0
                and math.isfinite(previous_best)
                and not nonfinite_val
                and val_loss > previous_best + loss_increase
            )
            if nonfinite_val or (min_step_met and (too_large or worsened)):
                if nonfinite_val:
                    stop_reason = "nonfinite_val_loss"
                else:
                    stop_reason = "val_loss_above_threshold" if too_large else "val_loss_regressed_from_best"
                stop_step = current_step
                stop_record = {
                    "event": "stopped_early",
                    "activation": args.activation,
                    "step": stop_step,
                    "reason": stop_reason,
                    "val_loss": val_loss,
                    "best_val_loss": None if not math.isfinite(best_val_loss) else best_val_loss,
                    "early_stop_max_val_loss": max_loss,
                    "early_stop_loss_increase": loss_increase,
                    "timing_attempt_id": timing_attempt_id,
                    "active_seconds_after_event": time.perf_counter() - start_time,
                }
                rank0_print(rank, json.dumps(stop_record, sort_keys=True))
                if rank == 0:
                    write_jsonl(out_path, stop_record)
                break

            checkpoint_interval = max(
                0, int(args.resume_checkpoint_interval)
            )
            if (
                resume_checkpoint is not None
                and checkpoint_interval > 0
                and current_step < args.steps
                and current_step % checkpoint_interval == 0
            ):
                checkpoint_active_seconds = (
                    time.perf_counter() - start_time
                )
                _save_exact_resume_checkpoint(
                    resume_checkpoint,
                    args=args,
                    rank=rank,
                    world_size=world_size,
                    is_distributed=is_distributed,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    train_generator=train_generator,
                    probe_state=probe_state,
                    completed_step=current_step,
                    best_val_loss=best_val_loss,
                    step_times=step_times,
                    optimizer_step_times=optimizer_step_times,
                    grad_clip_observed_steps=grad_clip_observed_steps,
                    grad_clip_triggered_steps=grad_clip_triggered_steps,
                    run_cuda_max_memory_allocated=(
                        run_cuda_max_memory_allocated
                    ),
                    run_cuda_max_memory_reserved=(
                        run_cuda_max_memory_reserved
                    ),
                    active_seconds=checkpoint_active_seconds,
                )
                if rank == 0:
                    write_jsonl(
                        out_path,
                        {
                            "event": "resume_checkpoint",
                            "activation": args.activation,
                            "step": current_step,
                            "path": str(
                                resume_checkpoint.resolve()
                            ),
                            "timing_attempt_id": timing_attempt_id,
                            "active_seconds_after_event": (
                                time.perf_counter() - start_time
                            ),
                        },
                    )

    total_time = time.perf_counter() - start_time
    warmup_drop = min(5, max(0, len(step_times) - 1))
    timed_steps = step_times[warmup_drop:]
    mean_step = sum(timed_steps) / max(1, len(timed_steps))
    completed_steps = stop_step if stop_step is not None else args.steps
    completed_tokens = int(completed_steps) * int(global_tokens)
    if device.type == "cuda":
        run_cuda_max_memory_allocated = max(
            run_cuda_max_memory_allocated,
            int(torch.cuda.max_memory_allocated(device)),
        )
        run_cuda_max_memory_reserved = max(
            run_cuda_max_memory_reserved,
            int(torch.cuda.max_memory_reserved(device)),
        )
        if dist.is_available() and dist.is_initialized():
            global_memory_peaks = torch.tensor(
                [run_cuda_max_memory_allocated, run_cuda_max_memory_reserved],
                device=device,
                dtype=torch.int64,
            )
            dist.all_reduce(global_memory_peaks, op=dist.ReduceOp.MAX)
            run_cuda_max_memory_allocated = int(global_memory_peaks[0].item())
            run_cuda_max_memory_reserved = int(global_memory_peaks[1].item())
    summary = {
        "event": "summary",
        "activation": args.activation,
        "slurm_job_id": slurm_job_id,
        "slurm_restart_count": slurm_restart_count,
        "slurm_node": slurm_node,
        "timing_attempt_id": timing_attempt_id,
        "timing_guard_max_seconds_per_step": args.timing_guard_max_seconds_per_step,
        "timing_guard_max_optimizer_step_seconds": (
            args.timing_guard_max_optimizer_step_seconds
        ),
        "timing_guard_min_step": args.timing_guard_min_step,
        "mean_seconds_per_step": mean_step,
        "tokens_per_second": global_tokens / mean_step,
        "training_loop_tokens_per_second": completed_tokens / max(total_time, 1.0e-12),
        "total_seconds": total_time,
        "steps": args.steps,
        "completed_steps": completed_steps,
        "completed_tokens": completed_tokens,
        "realized_lr_audit_steps": realized_lr_audit_steps,
        "realized_lr_trace_sha256": (
            realized_lr_trace.hexdigest()
            if args.fairness_contract == EXACT_LR_WD_CONTRACT
            else None
        ),
        "grad_clip_observed_steps": grad_clip_observed_steps,
        "grad_clip_triggered_steps": grad_clip_triggered_steps,
        "grad_clip_trigger_fraction": (
            0.0
            if grad_clip_observed_steps == 0
            else float(grad_clip_triggered_steps) / float(grad_clip_observed_steps)
        ),
        "cuda_run_max_memory_allocated": (
            run_cuda_max_memory_allocated if device.type == "cuda" else None
        ),
        "cuda_run_max_memory_reserved": (
            run_cuda_max_memory_reserved if device.type == "cuda" else None
        ),
        "cuda_run_memory_peak_scope": (
            "global_max_all_ranks" if device.type == "cuda" else None
        ),
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
    }
    rank0_print(rank, json.dumps(summary, sort_keys=True))
    if rank == 0:
        write_jsonl(out_path, summary)
    if args.save_checkpoint and rank == 0:
        checkpoint_dir = (
            Path(args.checkpoint_dir)
            if args.checkpoint_dir is not None
            else Path(args.output_dir) / args.run_name / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        checkpoint_path = checkpoint_dir / f"{sanitize_name(args.activation)}_seed{args.seed}.pt"
        torch.save(
            {
                "model": raw_model.state_dict(),
                "args": vars(args),
                "config": config_record,
                "summary": summary,
                "param_count": param_count,
            },
            checkpoint_path,
        )
        checkpoint_record = {
            "event": "checkpoint",
            "activation": args.activation,
            "path": str(checkpoint_path),
            "seed": args.seed,
        }
        rank0_print(rank, json.dumps(checkpoint_record, sort_keys=True))
        write_jsonl(out_path, checkpoint_record)
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()

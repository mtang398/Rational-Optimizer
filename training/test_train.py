import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_PACKAGE = ROOT / "activation"
if str(ACTIVATION_PACKAGE) not in sys.path:
    sys.path.insert(0, str(ACTIVATION_PACKAGE))

from training import train


def _model_args(activation, *, scale="tiny"):
    if scale == "12l_768d":
        layers, d_model, heads, ffn_dim, seq_len = 12, 768, 12, 2048, 256
    elif scale == "18l_1024d":
        layers, d_model, heads, ffn_dim, seq_len = 18, 1024, 16, 3072, 256
    elif scale == "tiny":
        layers, d_model, heads, ffn_dim, seq_len = 2, 32, 4, 64, 8
    else:
        raise ValueError(scale)
    return SimpleNamespace(
        layers=layers,
        d_model=d_model,
        heads=heads,
        ffn_dim=ffn_dim,
        seq_len=seq_len,
        activation=activation,
        rational_init="silu",
        rational_group_size=16 if scale == "tiny" else 256,
        rational_max_groups=8 if scale == "tiny" else 32,
        rational_basis_eps=1e-6,
        init_std=0.02,
    )


def _cache_args(tmp_path):
    return SimpleNamespace(
        dataset_name="synthetic/arithmetic",
        dataset_config=None,
        dataset_revision="dataset-revision",
        dataset_streaming=False,
        dataset_text_column="text",
        train_split="train",
        validation_split="validation",
        train_skip_documents=0,
        validation_skip_documents=0,
        train_skip_tokens=0,
        validation_skip_tokens=0,
        tokenizer="test-tokenizer",
        tokenizer_revision="tokenizer-revision",
        cache_dir=str(tmp_path),
        hf_cache=None,
        refresh_cache=False,
        tokenize_batch_size=8,
        seq_len=16,
    )


class _TestTokenizer:
    eos_token_id = 7
    pad_token_id = None

    def __call__(self, texts, add_special_tokens=False):
        assert not add_special_tokens
        return {
            "input_ids": [
                [(ord(character) % 47) + 1 for character in text]
                for text in texts
            ]
        }


@pytest.mark.parametrize(
    ("scale", "activation", "expected"),
    [
        ("12l_768d", "silu", 123_551_232),
        ("12l_768d", train.GRAIN_ACTIVATION, 123_552_672),
        ("18l_1024d", "silu", 296_867_840),
        ("18l_1024d", train.GRAIN_ACTIVATION, 296_871_080),
    ],
)
def test_published_model_parameter_counts(scale, activation, expected):
    with torch.device("meta"):
        model = train.CausalTransformer(
            _model_args(activation, scale=scale), 50_257
        )
    assert sum(parameter.numel() for parameter in model.parameters()) == expected


@pytest.mark.parametrize("activation", train.ACTIVATIONS)
def test_same_seed_initial_state_forward_and_backward(activation, monkeypatch):
    monkeypatch.setenv("RATIONAL_OPT_TORCH_FALLBACK", "1")
    args = _model_args(activation)
    models = []
    for _ in range(2):
        torch.manual_seed(2026)
        models.append(train.CausalTransformer(args, 97))

    first_state = models[0].state_dict()
    second_state = models[1].state_dict()
    assert first_state.keys() == second_state.keys()
    assert all(
        torch.equal(first_state[name], second_state[name])
        for name in first_state
    )

    input_ids = torch.arange(16, dtype=torch.long).view(2, 8) % 97
    outputs = [model(input_ids) for model in models]
    assert torch.equal(outputs[0], outputs[1])
    for output in outputs:
        output.float().square().mean().backward()
    first_gradients = dict(models[0].named_parameters())
    second_gradients = dict(models[1].named_parameters())
    assert first_gradients.keys() == second_gradients.keys()
    assert all(
        torch.equal(
            first_gradients[name].grad,
            second_gradients[name].grad,
        )
        for name in first_gradients
    )


def test_tensor_content_sha256_hashes_logical_bytes():
    tensor = torch.arange(257, dtype=torch.int32).view(257, 1)[::2]
    expected = hashlib.sha256(
        np.ascontiguousarray(tensor.numpy()).view(np.uint8)
    ).hexdigest()
    assert train.tensor_content_sha256(tensor, chunk_bytes=17) == expected


def test_cache_roundtrip_is_revision_bound_and_self_describing(
    tmp_path, monkeypatch
):
    calls = []

    def tokenizer_factory(*args, **kwargs):
        calls.append((args, kwargs))
        return _TestTokenizer()

    monkeypatch.setattr(
        train.transformers.AutoTokenizer,
        "from_pretrained",
        tokenizer_factory,
    )
    args = _cache_args(tmp_path)
    tokens = train.load_or_tokenize(args, "train", 513)
    cache_path = train.token_cache_path(args, "train", 513)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)

    assert len(calls) == 1
    assert calls[0][1]["revision"] == args.tokenizer_revision
    assert payload["schema"] == "rationalopt_token_cache_v2"
    assert payload["dataset_revision"] == args.dataset_revision
    assert payload["tokenizer_revision"] == args.tokenizer_revision
    assert payload["token_count"] == tokens.numel() == 513
    assert payload["tokens"].dtype == torch.int32
    assert payload["token_content_sha256"] == train.tensor_content_sha256(
        payload["tokens"]
    )
    assert not list(tmp_path.glob(".*.tmp-*"))

    def fail_if_rehashed(*args, **kwargs):
        raise AssertionError("a cache hit must not rescan token bytes")

    monkeypatch.setattr(train, "tensor_content_sha256", fail_if_rehashed)
    assert torch.equal(
        train.load_or_tokenize(args, "train", 513), payload["tokens"]
    )
    assert len(calls) == 1


def test_cache_rejects_malformed_content_hash(tmp_path):
    args = _cache_args(tmp_path)
    cache_path = train.token_cache_path(args, "train", 513)
    tokens = torch.arange(513, dtype=torch.int32)
    torch.save(
        {
            "schema": "rationalopt_token_cache_v2",
            "tokens": tokens,
            "token_count": tokens.numel(),
            "token_content_sha256": "not-a-sha256",
            **train._cache_metadata(args, "train"),
        },
        cache_path,
    )
    with pytest.raises(RuntimeError, match="token_content_sha256"):
        train._load_cached_tokens(cache_path, args, "train", 513)


def test_public_activation_surface_and_sam_contract(monkeypatch):
    import rational_opt

    assert train.ACTIVATIONS == ("silu", "grain")
    assert rational_opt.__all__ == ["GRAIN", "rational_local_basis"]
    assert not hasattr(rational_opt, "RationalFusedGlobalA5_4")
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--activation", train.GRAIN_ACTIVATION],
    )
    args = train.parse_args()
    train.validate_optimizer_protocol(args)
    assert args.telemetry_rlb_stat_every == 4
    assert args.telemetry_rlb_stat_samples == 512
    assert args.rlb_init_gauge_log_scale == 0.0
    assert args.rlb_init_gauge_seed == 424242
    args.sam_rho = 0.1
    with pytest.raises(ValueError, match="SAM"):
        train.validate_optimizer_protocol(args)

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--activation", "rlb_fused_global_rational"],
    )
    legacy_args = train.parse_args()
    train.validate_optimizer_protocol(legacy_args)


@pytest.mark.parametrize("activation", train.ACTIVATIONS)
@pytest.mark.parametrize(
    ("optimizer_name", "learning_rate"),
    [
        ("adamw", 3e-4),
        ("muon", 3e-4),
        ("lion", 1e-4),
        ("soap_adamw", 3e-4),
        ("ademamix", 3e-4),
        ("adafactor_came", 3e-4),
        ("schedule_free_adamw", 3e-4),
    ],
)
def test_exact_lr_wd_contract_is_row_aware(
    activation, optimizer_name, learning_rate, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--activation",
            activation,
            "--optimizer",
            optimizer_name,
            "--lr",
            str(learning_rate),
            "--min-lr",
            str(learning_rate / 10.0),
            "--weight-decay",
            "0.1",
            "--fairness-contract",
            train.EXACT_LR_WD_CONTRACT,
        ],
    )
    args = train.parse_args()
    args.layers = 2
    args.d_model = 32
    args.heads = 4
    args.ffn_dim = 64
    args.seq_len = 8
    args.rational_group_size = 16
    args.rational_max_groups = 8
    train.validate_optimizer_protocol(args)
    model = train.CausalTransformer(args, 97)
    optimizer = train.configure_optimizer(model, args)
    report = train.audit_optimizer_lr_wd_fairness(model, optimizer, args)
    assert report["contract"] == train.EXACT_LR_WD_CONTRACT
    assert report["passed"] is True
    assert report["base_lr"] == learning_rate
    assert report["minimum_lr"] == learning_rate / 10.0
    assert report["base_weight_decay"] == 0.1
    assert report["covered_parameter_elements"] == report["trainable_parameter_elements"]

    scheduled_lr = learning_rate * 0.5
    for group in optimizer.param_groups:
        group["lr"] = scheduled_lr
    train.assert_optimizer_realized_lr(optimizer, scheduled_lr, args)
    optimizer.param_groups[0]["lr"] = scheduled_lr * 2.0
    with pytest.raises(RuntimeError, match="realized LR mismatch"):
        train.assert_optimizer_realized_lr(optimizer, scheduled_lr, args)


def test_baseline_main_path_does_not_require_tiller_identity(monkeypatch):
    """The shared LR/WD audit must not turn a baseline into a TILLER run."""

    class _Tokens:
        def numel(self):
            return 4096

        def pin_memory(self):
            return self

    class _Tokenizer:
        def __len__(self):
            return 97

    class _Model:
        def state_dict(self):
            return {}

        def to(self, device):
            return self

        def parameters(self):
            return iter(())

        def named_parameters(self):
            return iter(())

    class _ReachedOptimizer(RuntimeError):
        pass

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--activation",
            "silu",
            "--optimizer",
            "lion",
            "--lr",
            "0.0001",
            "--min-lr",
            "0.00001",
            "--fairness-contract",
            train.EXACT_LR_WD_CONTRACT,
        ],
    )
    monkeypatch.setattr(
        train,
        "setup_distributed",
        lambda: (False, 0, 0, 1, torch.device("cuda")),
    )
    monkeypatch.setattr(train, "load_or_tokenize", lambda *args: _Tokens())
    monkeypatch.setattr(
        train.transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    monkeypatch.setattr(train, "CausalTransformer", lambda *args: _Model())
    monkeypatch.setattr(train.torch.cuda, "manual_seed_all", lambda seed: None)

    def reached_optimizer(*args):
        raise _ReachedOptimizer("baseline reached optimizer construction")

    monkeypatch.setattr(train, "configure_optimizer", reached_optimizer)
    monkeypatch.setattr(train, "_TILLER_IDENTITY_AUDITOR", None)
    with pytest.raises(_ReachedOptimizer, match="reached optimizer construction"):
        train.main()

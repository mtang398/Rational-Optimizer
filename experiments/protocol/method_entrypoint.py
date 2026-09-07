#!/usr/bin/env python3
"""Connect TILLER to the shared language-model trainer."""

from __future__ import annotations

import torch

from optimizer_design.tiller import (
    FAMILY_ID,
    TILLERAttentionOptimizer,
    TILLERRouter,
    configure_microbatch_count,
)
from training import train as trainer


OPTIMIZER_ID = FAMILY_ID
_BASE_CONFIGURE_OPTIMIZER = trainer.configure_optimizer
_BASE_CLIP_OR_MEASURE_GRADIENTS = trainer.clip_or_measure_gradients
_ACTIVE_ROUTER = None
_DECAY_TIED_EMBEDDING = False


class TILLERCompositeOptimizer(trainer.CompositeOptimizer):
    """Checkpointed composition of TILLER and ordinary AdamW parameters."""

    _ROLES = ("tiller_router", "tiller_attention", "adamw")
    _SCHEMA = "tiller_composite_v1"

    def __init__(self, optimizers):
        super().__init__(optimizers)
        if len(self.optimizers) != len(self._ROLES):
            raise RuntimeError("TILLER composite requires three optimizer children")

    def state_dict(self):
        return {
            "schema": self._SCHEMA,
            "roles": self._ROLES,
            "children": [child.state_dict() for child in self.optimizers],
        }

    def load_state_dict(self, state_dict):
        if (
            not isinstance(state_dict, dict)
            or state_dict.get("schema") != self._SCHEMA
            or tuple(state_dict.get("roles", ())) != self._ROLES
            or len(state_dict.get("children", ())) != len(self.optimizers)
        ):
            raise RuntimeError("TILLER composite checkpoint schema changed")
        for child, child_state in zip(self.optimizers, state_dict["children"]):
            child.load_state_dict(child_state)


def set_experiment_layout(*, microbatches: int, decay_tied_embedding: bool) -> None:
    """Install the model-scale layout encoded by one manifest row."""

    global _DECAY_TIED_EMBEDDING
    configure_microbatch_count(int(microbatches))
    _DECAY_TIED_EMBEDDING = bool(decay_tied_embedding)


def collect_blocks(model, args):
    """Collect the two MLP and two attention matrices from each block."""

    raw_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    groups = trainer.collect_rlb_optimizer_groups(model, args)
    by_layer = {int(group["layer_index"]): dict(group) for group in groups}
    if len(by_layer) != int(args.layers) or len(raw_model.layers) != int(args.layers):
        raise RuntimeError("TILLER requires one rational block per Transformer layer")
    blocks = []
    for layer_index, block in enumerate(raw_model.layers):
        group = by_layer.get(layer_index)
        if group is None or group["mlp"] is not block.mlp:
            raise RuntimeError("rational block inventory does not match model depth")
        group.update({
            "block": block,
            "qkv_weight": block.attn.qkv.weight,
            "attn_out_weight": block.attn.out.weight,
        })
        blocks.append(group)
    return blocks


def partition_parameters(model, blocks):
    """Assign each trainable tensor to TILLER or ordinary AdamW exactly once."""

    structural_ids = {
        id(parameter)
        for block in blocks
        for parameter in (
            block["in_weight"],
            block["out_weight"],
            block["qkv_weight"],
            block["attn_out_weight"],
        )
    }
    if len(structural_ids) != 4 * len(blocks):
        raise RuntimeError("TILLER structural parameter ownership overlaps")

    adam_decay = []
    adam_no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in structural_ids:
            continue
        tied = trainer.is_tied_embedding_parameter_name(name)
        no_decay = trainer.is_no_decay_parameter(name, parameter)
        if no_decay or (tied and not _DECAY_TIED_EMBEDDING):
            adam_no_decay.append(parameter)
        else:
            adam_decay.append(parameter)
    return structural_ids, adam_decay, adam_no_decay


def configure_candidate_optimizer(model, args):
    """Build the optimizer composition used by every TILLER manifest row."""

    global _ACTIVE_ROUTER
    if args.optimizer != OPTIMIZER_ID:
        return _BASE_CONFIGURE_OPTIMIZER(model, args)

    blocks = collect_blocks(model, args)
    structural_ids, adam_decay, adam_no_decay = partition_parameters(model, blocks)
    covered = structural_ids | {id(parameter) for parameter in adam_decay + adam_no_decay}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if covered != trainable:
        raise RuntimeError("TILLER parameter partition is incomplete")

    router = TILLERRouter(
        blocks,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
        ns_steps=args.muon_ns_steps,
        beta2=args.beta2,
        eps=args.eps,
    )
    attention = TILLERAttentionOptimizer(
        blocks,
        router,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
        ns_steps=args.muon_ns_steps,
        beta2=args.beta2,
        eps=args.eps,
        adjust_lr_fn=args.muon_adjust_lr_fn,
    )
    adam_groups = []
    if adam_decay:
        adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
    if not adam_groups:
        raise RuntimeError("TILLER found no parameters for its AdamW child")
    adamw = torch.optim.AdamW(
        adam_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    _ACTIVE_ROUTER = router
    return TILLERCompositeOptimizer((router, attention, adamw))


def clip_candidate_gradients(model, grad_clip, capture_norm):
    result = _BASE_CLIP_OR_MEASURE_GRADIENTS(model, grad_clip, capture_norm)
    if _ACTIVE_ROUTER is not None:
        if result[0] is None:
            raise RuntimeError("TILLER requires the realized gradient norm")
        _ACTIVE_ROUTER.record_realized_clipping(result[0], grad_clip)
    return result


def main():
    trainer.RATIONAL_SPECIFIC_OPTIMIZERS.add(OPTIMIZER_ID)
    trainer.RLB_MATRIX_SYNC_OPTIMIZERS.add(OPTIMIZER_ID)
    trainer.RLB_COEFFICIENT_SYNC_OPTIMIZERS.add(OPTIMIZER_ID)
    trainer.ACTIVE_OPTIMIZERS = sorted(set(trainer.ACTIVE_OPTIMIZERS) | {OPTIMIZER_ID})
    trainer.configure_optimizer = configure_candidate_optimizer
    trainer.clip_or_measure_gradients = clip_candidate_gradients
    trainer.main()


if __name__ == "__main__":
    main()

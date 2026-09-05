"""Grouped collective execution for the qualified fast Method1 parent.

The Method1 outer transaction independently reduces a score Fisher and its
weight-decay cross term.  NCCL applies the same elementwise SUM when those
two tensors are contiguous slices of one packet, while paying collective
latency once.  The local sample count is identical on every contracted rank,
so its all-reduce is replaced by the exactly representable product with the
world size.

No optimizer equation, refresh cadence, tensor precision, NS5 map, momentum,
budget, LR, or WD changes.  Packet size can nevertheless affect NCCL's
floating-point reduction schedule, so promotion as an exact rewrite requires
the production four-rank bitwise gate.  If that gate observes any drift, this
is a numerical execution candidate and needs a fresh complete quality run.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ._archive_r07_frame_878462.rlb_r01_core import R01Core as _ExactR01Core
from ._method1_metric2_approx.rlb_r01_core import _PATCH_LOCK
from .rlb_r07_frame_878462_metric2 import (
    Method1Metric2AttentionOptimizer,
    Method1Metric2Outer4Optimizer,
)


FAMILY_ID = "method1_878462_metric2_outer4_grouped_collectives_v1"
_EXACT_GLOBAL_REDUCER = _ExactR01Core._reduce_global_loss_metric


def _distributed() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def _all_reduce_pair_sum(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """SUM two tensors as one contiguous collective packet."""
    if not _distributed():
        return first, second
    first_count = first.numel()
    second_count = second.numel()
    packet = torch.cat((first.reshape(-1), second.reshape(-1)))
    dist.all_reduce(packet, op=dist.ReduceOp.SUM)
    return (
        packet[:first_count].view_as(first),
        packet[first_count:first_count + second_count].view_as(second),
    )


def _global_loss_metric_grouped(
    images: torch.Tensor,
    cotangents: torch.Tensor,
    group_decay_images: torch.Tensor,
):
    """Literal archived R01 statistic with one grouped SUM collective."""
    if images.ndim != 4 or cotangents.ndim != 3:
        raise RuntimeError("R01 global loss-metric tensor rank changed")
    layers, samples, groups, residual = images.shape
    if (
        cotangents.shape != (layers, samples, residual)
        or group_decay_images.shape != images.shape
    ):
        raise RuntimeError("R01 global loss-metric inventory changed")

    scores = torch.einsum(
        "lngd,lnd->nlg", images, cotangents
    ).reshape(samples, layers * groups)
    decay_scores = torch.einsum(
        "lngd,lnd->nlg", group_decay_images, cotangents
    ).sum(dim=(1, 2))
    fisher_sum = scores.transpose(0, 1) @ scores
    cross_sum = scores.transpose(0, 1) @ decay_scores
    count = torch.tensor(
        float(samples), device=images.device, dtype=images.dtype
    )
    if _distributed():
        fisher_sum, cross_sum = _all_reduce_pair_sum(fisher_sum, cross_sum)
        count.mul_(int(dist.get_world_size()))
    torch._assert_async(torch.isfinite(count) & (count > 0.0))
    fisher = fisher_sum / count
    fisher = 0.5 * (fisher + fisher.transpose(-2, -1))
    return fisher.unsqueeze(0), (cross_sum / count).unsqueeze(0), count


def _loss_metric_grouped(
    images: torch.Tensor,
    cotangents: torch.Tensor,
    common_decay_image: torch.Tensor,
):
    """Literal archived R09 statistic with one grouped SUM collective."""
    if images.ndim != 4 or cotangents.ndim != 3:
        raise RuntimeError("R09 loss-metric tensor rank changed")
    layers, samples, groups, residual = images.shape
    if (
        cotangents.shape != (layers, samples, residual)
        or common_decay_image.shape != (layers, samples, residual)
    ):
        raise RuntimeError("R09 loss-metric aligned inventory changed")
    scores = torch.einsum("lngd,lnd->lng", images, cotangents)
    decay_scores = torch.einsum(
        "lnd,lnd->ln", common_decay_image, cotangents
    )
    fisher_sum = torch.einsum("lng,lnv->lgv", scores, scores)
    cross_sum = torch.einsum("lng,ln->lg", scores, decay_scores)
    count = torch.tensor(
        float(samples), device=images.device, dtype=images.dtype
    )
    if _distributed():
        fisher_sum, cross_sum = _all_reduce_pair_sum(fisher_sum, cross_sum)
        count.mul_(int(dist.get_world_size()))
    torch._assert_async(torch.isfinite(count) & (count > 0.0))
    fisher = fisher_sum / count
    fisher = 0.5 * (fisher + fisher.transpose(-2, -1))
    return fisher, cross_sum / count, count


class _Method1GroupedCollectiveMixin:
    """Install grouped reductions only while this Method1 instance steps."""

    _grouped_family_id = FAMILY_ID
    _grouped_exact_r01_core = _ExactR01Core
    _grouped_exact_global_reducer = staticmethod(_EXACT_GLOBAL_REDUCER)
    _grouped_patch_lock = _PATCH_LOCK

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["rlb_grouped_collective_family_id"] = self._grouped_family_id
        group["rlb_grouped_fisher_cross_reduction"] = True
        group["rlb_deterministic_count_reduction"] = True

    _reduce_loss_metric = staticmethod(_loss_metric_grouped)

    def _consume_input_moments(self):
        if not self._capture_full_metric_this_step:
            return super()._consume_input_moments()
        samples = []
        for index, records in enumerate(self._input_records):
            if len(records) != self.expected_microbatches:
                raise RuntimeError(
                    f"R08 layer {index} did not observe four input microbatches"
                )
            combined = torch.cat(records, dim=0).float()
            self._input_records[index] = []
            expected = self.input_capture_count * self.expected_microbatches
            if combined.shape != (expected, self.external_width):
                raise RuntimeError("R08 input sample inventory changed")
            samples.append(combined)
        sample_batch = torch.stack(samples)
        moments = torch.bmm(sample_batch.transpose(1, 2), sample_batch)
        counts = torch.full(
            (len(self.pairs),),
            float(sample_batch.shape[1]),
            device=sample_batch.device,
            dtype=sample_batch.dtype,
        )
        if _distributed():
            dist.all_reduce(moments, op=dist.ReduceOp.SUM)
            counts.mul_(int(dist.get_world_size()))
        moments.div_(counts[:, None, None])
        self._last_input_record_count = self.expected_microbatches
        self._last_input_local_sample_count = int(sample_batch.shape[1])
        return moments, counts

    @torch.no_grad()
    def step(self, closure=None):
        # R03 deliberately wraps the archived R01 reducer.  Patching that one
        # descriptor under the overlay's existing re-entrant lock preserves
        # the complete R03 recurrence without copying or weakening it.
        exact_core = self._grouped_exact_r01_core
        with self._grouped_patch_lock:
            if exact_core._reduce_global_loss_metric is not (
                self._grouped_exact_global_reducer
            ):
                raise RuntimeError("RLB global reducer was already patched")
            exact_core._reduce_global_loss_metric = staticmethod(
                _global_loss_metric_grouped
            )
            try:
                return super().step(closure)
            finally:
                exact_core._reduce_global_loss_metric = staticmethod(
                    self._grouped_exact_global_reducer
                )

    def grouped_collective_runtime_report(self):
        return {
            "family_id": self._grouped_family_id,
            "global_fisher_cross_collectives_before": 3,
            "global_fisher_cross_collectives_after": 1,
            "layer_fisher_cross_collectives_before": 3,
            "layer_fisher_cross_collectives_after": 1,
            "input_metric_collectives_before": 2,
            "input_metric_collectives_after": 1,
            "deterministic_count_product_exact": True,
            "scientific_equations_changed": False,
            "refresh_cadence_changed": False,
            "ns5_changed": False,
            "lr_or_wd_changed": False,
            "production_four_rank_bitwise_gate_required": True,
            "fresh_quality_required_if_any_bitwise_drift": True,
        }


class Method1GroupedCollectiveOptimizer(
    _Method1GroupedCollectiveMixin,
    Method1Metric2Outer4Optimizer,
):
    checkpoint_schema = FAMILY_ID


__all__ = (
    "FAMILY_ID",
    "Method1GroupedCollectiveOptimizer",
    "Method1Metric2AttentionOptimizer",
)

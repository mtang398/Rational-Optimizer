"""Periodic full-metric refresh for the R01 row-polar execution path.

This opt-in approximation evaluates the complete observed outgoing-feature
and residual-input coordinate metrics once every eight optimizer steps and
reuses the most recently evaluated factors in between.  The refresh step is
the literal full-metric row-polar transaction.  Response routing, aligned
functional samples, the 324-coordinate Fisher solve, momentum, budgets, LR,
and WD remain active on every step.

It is a numerical method variant and requires a fresh quality trajectory.
The exact archive and the full-metric row implementation remain untouched.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_r01_9150_archive import verify_r01_9150_archive
from .rlb_r01_9150_cheap_polar import (
    R01RowPolarOptimizer,
    R02RowPolarAttentionOptimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
STALE_METRIC_FAMILY_ID = "r01_9150_periodic_full_metric_row_polar_v1"
METRIC_REFRESH_INTERVAL = 8


class _PeriodicFullMetricMixin:
    def __init__(self, pairs, **kwargs):
        pairs = list(pairs)
        self._metric_refresh_step = 0
        self._capture_full_metric_this_step = True
        self._cached_metric_factors = {}
        self._stale_input_record_counts = [0 for _ in pairs]
        self._stale_feature_record_counts = [0 for _ in pairs]
        super().__init__(pairs, **kwargs)
        self.param_groups[0]["r01_metric_refresh_interval"] = (
            METRIC_REFRESH_INTERVAL
        )

    def _make_input_hook(self, index):
        exact_capture = super()._make_input_hook(index)

        @torch.no_grad()
        def capture(module, inputs):
            if self._capture_full_metric_this_step:
                exact_capture(module, inputs)
                return
            if not module.training:
                return
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError("stale-metric input hook received invalid input")
            if self._functional_pending_inputs[index] is not None:
                raise RuntimeError("stale-metric input was not paired with a response")
            flat = inputs[0].detach().reshape(-1, self.external_width)
            indices = self._functional_row_indices(flat.shape[0], flat.device)
            self._functional_pending_inputs[index] = (
                int(flat.shape[0]),
                indices,
                flat.index_select(0, indices),
            )
            self._stale_input_record_counts[index] += 1

        return capture

    def _make_feature_hook(self, index):
        exact_capture = super()._make_feature_hook(index)

        @torch.no_grad()
        def capture(module, inputs, output):
            if self._capture_full_metric_this_step:
                exact_capture(module, inputs, output)
                return
            if not module.training:
                return
            pending = self._functional_pending_inputs[index]
            if pending is None:
                raise RuntimeError("stale-metric response has no aligned input")
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError("stale-metric response hook received invalid input")
            if not torch.is_tensor(output):
                raise RuntimeError("stale-metric response hook received invalid output")
            row_count, indices, sampled_input = pending
            flat_z = inputs[0].detach().reshape(-1, self.hidden)
            flat_h = output.detach().reshape(-1, self.hidden)
            if flat_z.shape[0] != row_count or flat_h.shape[0] != row_count:
                raise RuntimeError("stale-metric x/z/h token inventories differ")
            self._functional_records[index].append(
                (
                    sampled_input,
                    flat_z.index_select(0, indices),
                    flat_h.index_select(0, indices),
                )
            )
            self._functional_pending_inputs[index] = None
            self._stale_feature_record_counts[index] += 1

        return capture

    def _consume_feature_moment(self, layer_index):
        if self._capture_full_metric_this_step:
            return super()._consume_feature_moment(layer_index)
        count = self._stale_feature_record_counts[layer_index]
        self._stale_feature_record_counts[layer_index] = 0
        if count != self.expected_microbatches:
            raise RuntimeError("stale-metric feature inventory changed")
        if self._feature_moment_sums[layer_index] is not None:
            raise RuntimeError("stale-metric unexpectedly formed a feature Gram")
        sample_count = self.feature_capture_count * count
        self._last_feature_record_count = count
        self._last_feature_sample_count = sample_count
        marker = torch.ones(
            (1, 1, 1),
            device=self.incoming[0].device,
            dtype=torch.float32,
        )
        return marker, sample_count

    def _consume_input_moments(self):
        if self._capture_full_metric_this_step:
            return super()._consume_input_moments()
        for index, count in enumerate(self._stale_input_record_counts):
            if count != self.expected_microbatches:
                raise RuntimeError(
                    f"stale-metric layer {index} observed {count} input microbatches"
                )
        self._stale_input_record_counts = [0 for _ in self.pairs]
        if any(records for records in self._input_records):
            raise RuntimeError("stale-metric unexpectedly retained input rows")
        local_count = self.input_capture_count * self.expected_microbatches
        world = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        marker = torch.ones(
            (len(self.pairs), 1, 1),
            device=self.incoming[0].device,
            dtype=torch.float32,
        )
        counts = torch.full(
            (len(self.pairs),),
            float(local_count * world),
            device=marker.device,
            dtype=marker.dtype,
        )
        self._last_input_record_count = self.expected_microbatches
        self._last_input_local_sample_count = local_count
        return marker, counts

    def _unit_volume_cholesky(self, metric, *, capture_spectrum=False):
        call = self._r02_metric_factor_call
        if call not in (1, 2):
            return super()._unit_volume_cholesky(
                metric, capture_spectrum=capture_spectrum
            )
        if self._capture_full_metric_this_step:
            result = super()._unit_volume_cholesky(
                metric, capture_spectrum=capture_spectrum
            )
            self._cached_metric_factors[int(call)] = result
            return result
        if capture_spectrum:
            raise RuntimeError("stale-metric step cannot capture a fresh spectrum")
        cached = self._cached_metric_factors.get(int(call))
        if cached is None:
            raise RuntimeError("stale-metric factor was not initialized")
        self._r02_metric_factor_call = int(call) + 1
        return cached

    @torch.no_grad()
    def step(self, closure=None):
        refresh = self._capture_full_metric_this_step
        result = super().step(closure)
        self._metric_refresh_step += 1
        self._capture_full_metric_this_step = (
            self._metric_refresh_step % METRIC_REFRESH_INTERVAL == 0
        )
        if refresh and set(self._cached_metric_factors) != {1, 2}:
            raise RuntimeError("stale-metric refresh did not cache both factors")
        return result

    def periodic_metric_runtime_report(self):
        return {
            "family_id": STALE_METRIC_FAMILY_ID,
            "metric_refresh_interval": METRIC_REFRESH_INTERVAL,
            "refresh_step_is_literal_full_metric_row": True,
            "response_router_every_step": True,
            "functional_fisher_every_step": True,
            "polar_mode": "row_normalized",
            "lr_or_wd_changed": False,
        }


class R01StaleMetric8RowOptimizer(
    _PeriodicFullMetricMixin,
    R01RowPolarOptimizer,
):
    pass


R02StaleMetric8RowAttentionOptimizer = R02RowPolarAttentionOptimizer


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "METRIC_REFRESH_INTERVAL",
    "R01StaleMetric8RowOptimizer",
    "R02StaleMetric8RowAttentionOptimizer",
    "STALE_METRIC_FAMILY_ID",
)

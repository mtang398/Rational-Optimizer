"""Stdlib-only frozen matched-prefix contract for the switch experiment."""
import copy

CONDITION = 'grain_tiller100_muon'
SWITCH_UPDATE = 100
TARGET_UPDATES = 3815
SCHEDULE_HORIZON = 45776
POLICY = {
    'version': 'tiller100_then_native_muon_v1',
    'tiller_updates': SWITCH_UPDATE,
    'tiller_beta2': 0.95,
    'functional_probe_interval_steps': 8,
    'functional_probe_count': 32,
    'sketch_rank': 64,
    'tiller_variant': 'original_grain_tiller',
    'momentum_transfer': 'exact; unpack logical QKV into native Q/K/V',
    'auxiliary': 'retain AdamW object, moments, counters and hyperparameters',
    'model': 'GRAIN throughout; no activation or parameter conversion',
}

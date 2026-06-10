# ICLR Method Draft

This folder contains an Overleaf-style ICLR draft for RationalOPT.

The draft should use the manifest-first matched runs described in `../../experiments/ICLR_EXACT_RUN_PLAN.md`. Current E1 tables/curves and completed E2 DCLM M0/300M results live in `../../experiments/ICLR_RUN_STATUS.md`; the E2 result package is `../../experiments/results/iclr26_e2_dclm_2026_06_10/`; exact submitted commands live in `../../experiments/ICLR_RUN_COMMANDS.md`. WikiText may remain a small demo anchor when useful.

## Current Paper Position

```text
abstract: placeholder
introduction: placeholder
background and related work: placeholder
method: current substantive section
experiments: placeholder text; data source is current E1 status plus E2 DCLM package
discussion: placeholder
conclusion: placeholder
proof appendix: placeholder until the empirical claim and theory are stable
```

## Build

Use the included ICLR style files and compile `main.tex`. The local renderer used in this workspace is:

```bash
/home/mt872/autoresearch_attempt_1/.local/bin/tectonic main.tex
```

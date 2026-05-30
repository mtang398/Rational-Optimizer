# ICLR Method Draft

This folder is an Overleaf-ready ICLR paper draft. It contains a full visible paper skeleton with blank abstract, introduction, related work, experiments, discussion, limitations, and conclusion sections. The main paper contains a compact method section; lower-level update equations are placed in the appendix after the references, matching the ICLR author-guide placement for supplementary text.

## Format

As of May 30, 2026, I could not find a public ICLR 2027 template/style file. This folder therefore uses the official ICLR 2026 LaTeX style files linked from the ICLR 2026 Author Guide:

```text
https://iclr.cc/Conferences/2026/AuthorGuide
https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip
```

Included style files:

```text
iclr2026_conference.sty
iclr2026_conference.bst
math_commands.tex
fancyhdr.sty
natbib.sty
```

Switch this folder to the official ICLR 2027 template as soon as `iclr2027_conference.sty` or an official `iclr2027.zip` becomes available.

## Writing Reference

The method format follows the structure of an ICLR optimizer paper: formal setup, concise main update rule, and an algorithm box, with implementation-level equations moved to the appendix. The concrete reference checked for structure and citation metadata was Kingma and Ba's ICLR 2015 Adam paper:

```text
https://mlanthology.org/iclr/2015/kingma2015iclr-adam/
```

## Build

Local renderer used here:

```bash
/home/mt872/autoresearch_attempt_1/.local/bin/tectonic main.tex
```

For Overleaf, upload this folder and compile `main.tex` with the included style files.

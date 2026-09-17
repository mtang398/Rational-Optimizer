# ICLR 2027 paper

This directory is a self-contained ICLR 2027 manuscript project. It includes
the official conference style, bibliography style, and bundled compatibility
files, together with the paper source and a rendered PDF.

Read the [paper](main.pdf) or edit [the main text](main.tex) and
[appendix](appendix.tex). The paper develops GRAIN activations and TILLER
optimization for Transformer training. The main text introduces the
response-guided update construction and its full-horizon and staged use.
The appendix derives the activation geometry, matrix directions, loss
ledger, and coefficient selection, and presents both schedules in one
algorithm.

## Compile locally

The included PDF was built with pdfLaTeX and TeX Live 2026:

```bash
make
```

The build runs `latexmk -pdf` on `main.tex`. Add the TeX Live 2026
binary directory to `PATH` when it is not already available:

```bash
PATH=/path/to/texlive/2026/bin/x86_64-linux:$PATH make
```

## Compile on Overleaf

Upload the contents of this directory, set `main.tex` as the main document,
select pdfLaTeX, and select
[TeX Live 2026](https://www.overleaf.com/blog/tex-live-2026-is-now-available).
These are the compiler settings used for the included PDF. The bundled
conference files are
copied from the official ICLR 2027 style archive recorded in
`OFFICIAL_TEMPLATE.sha256`.

The official archive is
`https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`, with
SHA-256 digest
`0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`.
Verify the bundled files with:

```bash
sha256sum -c OFFICIAL_TEMPLATE.sha256
```

The project uses the anonymous review mode of the official style.
The [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
allow nine main-text pages for the initial submission, with references
and appendices following the main text.

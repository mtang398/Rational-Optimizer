# ICLR 2027 paper

This directory is a self-contained ICLR 2027 manuscript project. It includes
the official conference style, bibliography style, and bundled compatibility
files, together with the paper source and a rendered PDF.

Read the [paper](main.pdf) or edit [the LaTeX source](main.tex). The method
section introduces GRAIN's learned group responses and TILLER's construction
and coordination of update directions. The appendices develop the response
calculus, direction geometry, loss measurements, spectral approximation,
coefficient solver, and complete update algorithm.

## Compile locally

Use pdfLaTeX with TeX Live 2025, matching the Overleaf project settings:

```bash
make
```

The build runs `latexmk -pdf` on `main.tex`. Set `LATEXMK` when the TeX Live
2025 executable is not already on `PATH`:

```bash
make LATEXMK=/path/to/texlive/2025/bin/x86_64-linux/latexmk
```

## Compile on Overleaf

Upload the contents of this directory, set `main.tex` as the main document,
select pdfLaTeX, and select TeX Live 2025. The bundled conference files are
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

The submission switch `\iclrfinalcopy` remains disabled for anonymous review.

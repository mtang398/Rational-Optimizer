LATEXMK ?= latexmk

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib iclr2027_conference.sty iclr2027_conference.bst
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex

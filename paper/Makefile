LATEXMK ?= latexmk

.PHONY: all clean

all: main.pdf

main.pdf: main.tex appendix.tex references.bib math_commands.tex iclr2027_conference.sty iclr2027_conference.bst
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex

# Minimal makefile for Sphinx documentation.
#
# The published site is built by .github/workflows/docs.yml, which uses the
# same -W flag as the gate in ci.yml. Use `make strict` locally before
# pushing: a plain `make html` tolerates warnings that CI will reject.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html strict clean livehtml Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "  strict      build exactly as CI does, warnings treated as errors"
	@echo "  clean       remove _build, and the generated API and gallery trees"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

strict:
	@$(SPHINXBUILD) -b html -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" generated auto_examples sg_execution_times.rst

# Catch-all: route anything else to sphinx-build's -M mode.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

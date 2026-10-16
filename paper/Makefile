PYTHON ?= python3
TECTONIC ?= tectonic

.PHONY: all assets verify statistics pdf clean
all: pdf
assets:
	$(PYTHON) scripts/build_assets.py
	$(PYTHON) scripts/write_claim_matrix.py
verify:
	$(PYTHON) scripts/verify_evidence.py
statistics:
	$(PYTHON) scripts/verify_evidence.py --statistics
pdf: assets
	mkdir -p build
	$(TECTONIC) -X compile --outdir build --keep-logs --untrusted main.tex
	cp build/main.pdf main.pdf
clean:
	rm -rf build

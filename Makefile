.PHONY: charts

# Redraw every chart in docs/charts/ from the repo's own data. A test fails when a committed
# chart differs from what this writes.
charts:
	uv run --no-sync python bench/charts.py

.PHONY: install dataset format train eval all
install:
	python -m pip install -r requirements.txt
dataset:
	python download_dataset.py
format:
	python -m black --check train.py download_dataset.py
train: dataset
	python train.py
eval:
	python -c "from pathlib import Path; print(Path('results/metrics.txt').read_text())"
all: install train eval

from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
destination = DATA_DIR / "drug200.csv"

URLS = [
    "https://raw.githubusercontent.com/camara94/MachineLearningwithPython/main/drug200.csv",
    "https://raw.githubusercontent.com/kvinlazy/Dataset/master/drug200.csv",
]

if not destination.exists():
    last_error = None
    for url in URLS:
        try:
            print(f"Downloading dataset from {url}")
            urlretrieve(url, destination)
            break
        except Exception as exc:
            last_error = exc
            if destination.exists():
                destination.unlink()
    else:
        raise RuntimeError(f"Could not download dataset. Download drug200.csv manually into data/. Details: {last_error}")

print(f"Dataset ready: {destination}")

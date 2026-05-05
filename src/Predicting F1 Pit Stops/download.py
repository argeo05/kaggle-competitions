import kagglehub
import shutil
from pathlib import Path

cache_path = Path(kagglehub.competition_download("playground-series-s6e5"))
target_path = Path("./src/Predicting F1 Pit Stops/data/")

target_path.mkdir(parents=True, exist_ok=True)

for file in cache_path.iterdir():
    shutil.copy(file, target_path / file.name)

print("Saved to:", target_path)
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = ROOT / "reports/results.json"
if not json.loads(report.read_text()).get("summary"):
    raise ValueError("Finish evaluation before staging the demo")
shutil.copyfile(report, ROOT / "web/results.json")
print("Browser results updated. The API serves /artifacts from the checked-in artifacts directory.")

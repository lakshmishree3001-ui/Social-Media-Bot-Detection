"""Patch both notebooks to use robust path resolution."""
import json, re
from pathlib import Path

ROOT = Path("c:/Social Media Bot Detection")

NEW_SETUP_M2 = (
    "import sys, os, json\\n"
    "import numpy as np\\n"
    "import pandas as pd\\n"
    "import matplotlib.pyplot as plt\\n"
    "import matplotlib.gridspec as gridspec\\n"
    "import seaborn as sns\\n"
    "from pathlib import Path\\n"
    "\\n"
    "# -- Resolve project root for any Jupyter working directory\\n"
    "_ROOT = Path(os.path.abspath(''))\\n"
    "for _ in range(5):\\n"
    "    if (_ROOT / 'data' / 'raw').exists(): break\\n"
    "    _ROOT = _ROOT.parent\\n"
    "sys.path.insert(0, str(_ROOT))\\n"
    "RAW_DIR = _ROOT / 'data' / 'raw'\\n"
    "\\n"
    "pd.set_option('display.max_columns', 30)\\n"
    "pd.set_option('display.float_format', '{:.4f}'.format)\\n"
    "plt.rcParams['figure.dpi'] = 110\\n"
    "sns.set_theme(style='darkgrid', palette='muted')\\n"
    "print('RAW_DIR:', RAW_DIR.resolve())"
)

NEW_SETUP_M3 = (
    "import sys, os, json, warnings\\n"
    "warnings.filterwarnings('ignore')\\n"
    "\\n"
    "import numpy as np\\n"
    "import pandas as pd\\n"
    "import matplotlib.pyplot as plt\\n"
    "import seaborn as sns\\n"
    "from pathlib import Path\\n"
    "from sklearn.model_selection import train_test_split\\n"
    "from sklearn.preprocessing import LabelEncoder\\n"
    "\\n"
    "pd.set_option('display.max_columns', 30)\\n"
    "pd.set_option('display.float_format', '{:.4f}'.format)\\n"
    "plt.rcParams['figure.dpi'] = 110\\n"
    "sns.set_theme(style='darkgrid', palette='muted')\\n"
    "\\n"
    "# -- Resolve project root for any Jupyter working directory\\n"
    "_ROOT = Path(os.path.abspath(''))\\n"
    "for _ in range(5):\\n"
    "    if (_ROOT / 'data' / 'raw').exists(): break\\n"
    "    _ROOT = _ROOT.parent\\n"
    "sys.path.insert(0, str(_ROOT))\\n"
    "RAW_DIR       = _ROOT / 'data' / 'raw'\\n"
    "PROCESSED_DIR = _ROOT / 'data' / 'processed'\\n"
    "SPLITS_DIR    = _ROOT / 'data' / 'splits'\\n"
    "PROCESSED_DIR.mkdir(parents=True, exist_ok=True)\\n"
    "SPLITS_DIR.mkdir(parents=True, exist_ok=True)\\n"
    "\\n"
    "LABEL_MAP    = {0: 'human', 1: 'bot', 2: 'suspicious'}\\n"
    "LABEL_COLORS = {'human': '#4CAF50', 'bot': '#F44336', 'suspicious': '#FF9800'}\\n"
    "\\n"
    "print('All libraries loaded. RAW_DIR:', RAW_DIR.resolve())"
)

for nb_name, new_setup in [
    ("module2_dataset_understanding.ipynb", NEW_SETUP_M2),
    ("module3_data_preparation.ipynb",      NEW_SETUP_M3),
]:
    nb_path = ROOT / nb_name
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Find the first code cell (setup cell) and replace its source
    patched = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "sys.path" in src or "RAW_DIR" in src:
                # Rebuild source lines from new_setup
                cell["source"] = [
                    line + "\\n" if i < len(new_setup.split("\\n")) - 1 else line
                    for i, line in enumerate(new_setup.replace("\\n", "\n").split("\n"))
                ]
                patched = True
                break

    if patched:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Patched: {nb_name}")
    else:
        print(f"WARNING: Could not find setup cell in {nb_name}")

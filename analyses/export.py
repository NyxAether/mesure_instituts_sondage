import json
from pathlib import Path

import numpy as np

DOCS_DATA = Path(__file__).resolve().parent.parent / "docs" / "data"


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 6)
    return obj


def write_page_data(name: str, data: dict) -> Path:
    """Écrit les données d'une page dans `docs/data/<name>.js`.

    Un fichier JS (et non JSON) est utilisé pour que les pages s'ouvrent
    directement depuis le disque (file://), sans serveur ni fetch().
    """
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    path = DOCS_DATA / f"{name}.js"
    payload = json.dumps(_to_builtin(data), ensure_ascii=False, separators=(",", ":"))
    path.write_text(
        "// Fichier généré par analyses/ — ne pas modifier à la main.\n"
        f"window.DATA = window.DATA || {{}};\nwindow.DATA[{json.dumps(name)}] = {payload};\n",
        encoding="utf-8",
        newline="\n",  # LF sur tous les systèmes : sinon Windows écrit du CRLF et tout le fichier diffère
    )
    return path

"""Bootstrap comun pentru scripturile de evaluare.

Construieste un CerebrumAISystem IDENTIC cu cel din aplicatie (acelasi config,
DSN, Neo4j, model de embeddings), dar fara UI Streamlit. Refoloseste
`load_config` + `build_apci_config_dict` din main_flash ca sa nu duplicam logica.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

# Neutralizam set_page_config: e singurul apel la nivel de modul in main_flash
# care ar cere un context Streamlit. Asa putem importa functiile de config.
import streamlit as st  # noqa: E402

st.set_page_config = lambda *a, **k: None  # type: ignore

from main_flash import build_apci_config_dict, load_config  # noqa: E402
from rag_module_flash import create_apci_system  # noqa: E402

# Notebook-ul de evaluare (suprascrii cu env EVAL_NOTEBOOK daca ai alt nume).
NOTEBOOK = os.getenv("EVAL_NOTEBOOK", "Ai engineering Evaluation")

# Cele 4 configuratii (web neforțat la toate; controlul web e per-intrebare).
CONFIGS = {
    "A": {"enable_hyde": False, "enable_reranker": False, "enable_graph_rag": False},
    "B": {"enable_hyde": False, "enable_reranker": True, "enable_graph_rag": False},
    "C": {"enable_hyde": True, "enable_reranker": True, "enable_graph_rag": False},
    "D": {"enable_hyde": True, "enable_reranker": True, "enable_graph_rag": True},
}


def build_system():
    """Construieste si returneaza sistemul CerebrumAI (incarca torch/embeddings)."""
    config = load_config()
    config_dict = build_apci_config_dict(config)
    api_key = (os.getenv("GOOGLE_API_KEY", "") or config.get("gemini_api_key", "")).strip()
    return create_apci_system(api_key, config_dict)


def apply_config(system, name: str) -> None:
    """Seteaza flagurile configuratiei A/B/C/D si forteaza test_mode (fara memorie)."""
    flags = CONFIGS[name]
    system.config.enable_hyde = flags["enable_hyde"]
    system.config.enable_reranker = flags["enable_reranker"]
    system.config.enable_graph_rag = flags["enable_graph_rag"]
    system.config.test_mode = True  # evaluare retrieval: fara memorie personala

import json
from pathlib import Path

from loguru import logger

from src.models.corpus import BilingualCorpus


def load_corpus(path: Path) -> BilingualCorpus | None:
    """Load and validate one corpus JSON, warning and returning None on failure."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return BilingualCorpus.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to load {path.name}: {e}")
        return None

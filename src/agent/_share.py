from pathlib import Path
from typing import Final

PATTERN_CACHE_PATH: Final = Path("temp/session_patterns.json")

AUTO_APPROVE_THRESHOLD: Final = 95
MAX_TAG_RETRIES: Final = 2
DEFAULT_BATCH_SIZE: Final = 10
DEFAULT_NEARBY_RANGE: Final = 10
PATTERN_MIN_EXAMPLES: Final = 3
CONTEXT_COLLECTOR_CONCURRENCY: Final = 10

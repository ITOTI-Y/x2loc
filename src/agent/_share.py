from pathlib import Path
from typing import Final

PATTERN_CACHE_PATH: Final = Path("data/patterns.json")

MAX_TAG_RETRIES: Final = 2
DEFAULT_BATCH_SIZE: Final = 10
DEFAULT_NEARBY_RANGE: Final = 2
CONTEXT_COLLECTOR_CONCURRENCY: Final = 10
MAX_CONTEXT_COMPONENTS: Final = 6

# Pattern extraction
PATTERN_MIN_EXAMPLES: Final = 3
PATTERN_MAX_EXAMPLES: Final = 5
PATTERN_MAX_SOURCE_WORDS: Final = 24
MAX_MATCHES_PER_COMPONENT: Final = 3

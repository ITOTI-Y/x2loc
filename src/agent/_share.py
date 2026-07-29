from pathlib import Path
from typing import Final

PATTERN_CACHE_PATH: Final = Path("data/cache_translation_patterns.json")

MAX_TAG_RETRIES: Final = 2
DEFAULT_BATCH_SIZE: Final = 10
DEFAULT_NEARBY_RANGE: Final = 2
CONTEXT_COLLECTOR_CONCURRENCY: Final = 10
MAX_CONTEXT_COMPONENTS: Final = 6

# Total translate/validate/score rounds a batch may consume before the
# automatic path gives up and fails the job.
MAX_TRANSLATION_ATTEMPTS: Final = 3
DEFAULT_LLM_CONCURRENCY: Final = 10
MAX_LLM_CONCURRENCY: Final = 50

# Interactive sessions run one batch per astream segment (each interrupt
# starts a new invocation), so this fixed cap only has to cover a single
# segment. The automatic path translates a whole component in one invoke
# and computes its own bound via graph_recursion_limit().
GRAPH_RECURSION_LIMIT: Final = 500

# Pattern extraction
PATTERN_MIN_EXAMPLES: Final = 3
PATTERN_MAX_EXAMPLES: Final = 5
PATTERN_MAX_SOURCE_WORDS: Final = 24
MAX_MATCHES_PER_COMPONENT: Final = 3

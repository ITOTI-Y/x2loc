from typing import Final

DEFAULT_NEARBY_RANGE: Final = 2
CONTEXT_COLLECTOR_CONCURRENCY: Final = 10
CONTEXT_SEARCH_TIMEOUT: Final = 10.0
CONTEXT_SEARCH_ATTEMPTS: Final = 5
MAX_CONTEXT_COMPONENTS: Final = 6

# Total translate/validate/score rounds a batch may consume before the
# automatic path gives up and fails the job.
MAX_TRANSLATION_ATTEMPTS: Final = 3

# Interactive sessions run one batch per astream segment (each interrupt
# starts a new invocation), so this fixed cap only has to cover a single
# segment. The automatic path translates a whole component in one invoke
# and computes its own bound via graph_recursion_limit().
GRAPH_RECURSION_LIMIT: Final = 500

# Pattern extraction
PATTERN_MIN_EXAMPLES: Final = 3
PATTERN_LITERAL_MAJORITY: Final = 0.8
PATTERN_MAX_EXAMPLES: Final = 5
PATTERN_MAX_SOURCE_WORDS: Final = 24
MAX_MATCHES_PER_COMPONENT: Final = 3

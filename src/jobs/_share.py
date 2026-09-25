from typing import Final

# v1 fixes these. They are constants rather than settings fields because a
# config knob whose only legal value is its default is not a knob.
JOB_CONCURRENCY: Final = 1
SSE_PING_SECONDS: Final = 15
# Bounds how stale the base and mods glossaries may get from edits made in
# the Weblate UI; the service's own custom-glossary writes invalidate at once.
GLOSSARY_TTL_SECONDS: Final = 600

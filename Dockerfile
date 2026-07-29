FROM python:3.13-slim

# SteamCMD is a 32-bit binary; the tarball self-updates in place under
# /opt/steamcmd, which compose therefore mounts as a persistent volume.
RUN dpkg --add-architecture i386 \
    && apt-get update \
    && apt-get install -y --no-install-recommends lib32gcc-s1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/steamcmd \
    && curl -fsSL https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz \
        | tar -xz -C /opt/steamcmd

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Container conventions: paths live inside the container, the bind is
# 0.0.0.0 within the private network namespace, and exposure is decided
# by the host-side publish rule in compose. Everything else (credentials,
# port, glossary slugs) comes from the mounted configs/weblate.local.toml.
ENV X2LOC_STEAM__EXECUTABLE=/opt/steamcmd/steamcmd.sh \
    X2LOC_STEAM__ROOT=/srv/steam \
    X2LOC_DATA_ROOT=/srv/data \
    X2LOC_BIND_HOST=0.0.0.0 \
    X2LOC_ALLOW_NON_LOOPBACK_BIND=true

CMD ["/app/.venv/bin/x2loc-api"]

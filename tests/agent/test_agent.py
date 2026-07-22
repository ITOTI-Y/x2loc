from pathlib import Path

from joblib import Memory

from src.agent.cli import run

memory = Memory(location="data/_dev/cache")


@memory.cache
def _run_agent():
    config_path = Path("configs/weblate.local.toml")
    run(config_path=config_path, batch_size=10, auto_accept=False)


if __name__ == "__main__":
    _run_agent()

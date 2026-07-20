import asyncio
from pathlib import Path
from pickle import load

from src.agent.config import load_config
from src.agent.nodes.uploader import BackgroundUploader, uploader
from src.services.weblate import AsyncWeblateClient


async def debug_uploader() -> None:
    config = load_config("./configs/weblate.local.toml")
    client = AsyncWeblateClient(config.weblate)
    with Path("./data/_dev/uploader.pkl").open("rb") as f:
        state = load(f)
    background_uploader = BackgroundUploader(client, config)
    await uploader(state, background_uploader=background_uploader)
    await background_uploader.drain()
    await client.close()

if __name__ == "__main__":
    asyncio.run(debug_uploader())

import asyncio
import sys

from src.models.weblate import WeblateComponentSchema, WeblateLanguageSchema

sys.path.append("..")

from src.agent.config import load_config
from src.services.weblate import AsyncWeblateClient

config = load_config()


async def _test_weblate_client():
    async with AsyncWeblateClient(config.weblate) as client:
        project = await client.get_project()
        print(project)


async def _test_component():
    async with AsyncWeblateClient(config.weblate) as client:
        component = WeblateComponentSchema(
            name="Test Component",
            slug="test-component",
            source_csv=b"name,email\nJohn Doe,john.doe@example.com\nJane Doe,jane.doe@example.com",
            source_language=WeblateLanguageSchema(id=1, code="en", name="English"),
        )
        await client.list_components()
        await client.create_component(component=component)
        await client.delete_component(component_slug=component.slug)


async def _test_units():
    async with AsyncWeblateClient(config.weblate) as client:
        await client.list_units(
            component_slug="glossary-bases",
            lang="en",
            q="",
        )


async def test_all():
    await _test_weblate_client()
    await _test_component()
    await _test_units()


if __name__ == "__main__":
    asyncio.run(test_all())

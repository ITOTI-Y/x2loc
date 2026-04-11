from src.models._share import BaseSchema


class WeblateConfigSchema(BaseSchema):
    url: str
    token: str
    project_slug: str
    # SPDX identifier (e.g. "CC-BY-4.0", "MIT"). When set, every component
    # we create is marked with this license — suppresses Weblate's
    # "public project needs a license" warning.
    license: str = ""
    license_url: str = ""

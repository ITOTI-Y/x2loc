from typing import Final

LANG_EXT_MAP: Final[dict[str, str]] = {
    "int": "en",
    "chn": "zh_Hans",
    "cht": "zh_Hant",
    "deu": "de",
    "esn": "es",
    "fra": "fr",
    "ita": "it",
    "jpn": "ja",
    "kor": "ko",
    "pol": "pl",
    "rus": "ru",
}

EXT_LANG_MAP: Final[dict[str, str]] = {v: k for k, v in LANG_EXT_MAP.items()}

SUPPORTED_ENCODINGS: Final[list[str]] = ["utf-16-le"]

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from common.errors import EvaiError

CHINESE_SCRIPT_DICTIONARY_DIR = Path(__file__).with_name("dictionary")
SIMPLIFIED_TO_TRADITIONAL_STAGE: tuple[str, ...] = ("STPhrases.txt", "STCharacters.txt")
TAIWAN_PHRASE_STAGE: tuple[str, ...] = ("TWPhrases.txt",)
TAIWAN_VARIANT_STAGE: tuple[str, ...] = ("TWVariantsPhrases.txt", "TWVariants.txt")
COMMENT_PREFIX = "#"


@dataclass(frozen=True)
class ConversionStage:
    mapping: dict[str, str]
    longest_key: int

    def convert(self, text: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(text):
            for length in range(min(self.longest_key, len(text) - index), 0, -1):
                replacement = self.mapping.get(text[index : index + length])
                if replacement is not None:
                    output.append(replacement)
                    index += length
                    break
            else:
                output.append(text[index])
                index += 1
        return "".join(output)


def read_dictionary(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise EvaiError(f"Chinese script dictionary not found: {path}")
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(COMMENT_PREFIX):
            continue
        key, _, values = line.partition("\t")
        if key and values:
            mapping.setdefault(key, values.split(" ", 1)[0])
    return mapping


def build_stage(file_names: tuple[str, ...], root: Path) -> ConversionStage:
    mapping: dict[str, str] = {}
    for file_name in file_names:
        for key, value in read_dictionary(root / file_name).items():
            mapping.setdefault(key, value)
    return ConversionStage(mapping, max(map(len, mapping)))


@lru_cache(maxsize=1)
def taiwan_traditional_stages(
    root: Path = CHINESE_SCRIPT_DICTIONARY_DIR,
) -> tuple[ConversionStage, ...]:
    return tuple(
        build_stage(stage, root)
        for stage in (SIMPLIFIED_TO_TRADITIONAL_STAGE, TAIWAN_PHRASE_STAGE, TAIWAN_VARIANT_STAGE)
    )


def to_taiwan_traditional(text: str) -> str:
    for stage in taiwan_traditional_stages():
        text = stage.convert(text)
    return text

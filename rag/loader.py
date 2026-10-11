import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SPACES = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Document:
    name: str
    text: str


def clean_text(raw: str) -> str:
    lines = (_SPACES.sub(" ", line).strip() for line in raw.splitlines())
    return "\n".join(line for line in lines if line)


def load_documents(folder: Path) -> list[Document]:
    if not folder.is_dir():
        raise FileNotFoundError(f"not a directory: {folder}")
    docs = []
    for path in sorted(folder.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        text = clean_text(raw)
        if not text:
            logger.warning("skip %s: empty after cleaning", path.name)
            continue
        docs.append(Document(name=path.name, text=text))
        logger.debug("loaded %s (%d chars)", path.name, len(text))
    return docs

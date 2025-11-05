from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

from ..utils.logger import get_logger


MemoryKind = Literal["scratchpad", "plan"]


class MemoryManager:
    def __init__(self, scratchpad_path: Path, plan_path: Path) -> None:
        self.paths: Dict[MemoryKind, Path] = {
            "scratchpad": scratchpad_path,
            "plan": plan_path,
        }
        for path in self.paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("# Initialized\n", encoding="utf-8")
        self.logger = get_logger(self.__class__.__name__)

    def read(self, kind: MemoryKind) -> str:
        return self.paths[kind].read_text(encoding="utf-8")

    def write(self, kind: MemoryKind, content: str) -> None:
        self.paths[kind].write_text(content, encoding="utf-8")
        self.logger.info("Updated %s memory at %s", kind, self.paths[kind])

    def append(self, kind: MemoryKind, fragment: str) -> None:
        with self.paths[kind].open("a", encoding="utf-8") as file:
            file.write("\n")
            file.write(fragment)
        self.logger.debug("Appended to %s memory at %s", kind, self.paths[kind])

    def snapshot(self) -> Dict[MemoryKind, str]:
        return {kind: self.read(kind) for kind in self.paths}

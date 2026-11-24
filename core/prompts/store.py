"""Where prompts come from.

Prompts are files named ``<name>.v<version>.md``. Keeping every version on disk makes a
prompt change a reviewable diff and a rollback a one-line pin.
"""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from core.prompts.errors import PromptError, PromptNotFoundError
from core.prompts.types import Prompt

_FILENAME = re.compile(r"^(?P<name>[a-z][a-z0-9_]*)\.v(?P<version>\d+)\.md$")


class PromptStore(Protocol):
    def get(self, name: str) -> Prompt:
        """Return the active version of ``name`` or raise ``PromptNotFoundError``."""
        ...


class FilePromptStore:
    """Prompts in one folder. The newest version is active unless ``pinned`` says otherwise."""

    def __init__(self, directory: Path, pinned: Mapping[str, int] | None = None) -> None:
        self._directory = directory
        self._pinned = dict(pinned or {})
        self._versions: dict[str, dict[int, Path]] = {}
        if not directory.is_dir():
            raise PromptError(f"prompt folder not found: {directory}")
        for path in sorted(directory.iterdir()):
            if match := _FILENAME.match(path.name):
                versions = self._versions.setdefault(match["name"], {})
                versions[int(match["version"])] = path
        for name, version in self._pinned.items():
            if version not in self._versions.get(name, {}):
                raise PromptError(
                    f"prompt '{name}' is pinned to v{version}, but {directory} has "
                    f"versions {sorted(self._versions.get(name, {}))}"
                )

    def get(self, name: str) -> Prompt:
        versions = self._versions.get(name)
        if not versions:
            raise PromptNotFoundError(
                f"no prompt named '{name}' in {self._directory} (expected a file like {name}.v1.md)"
            )
        version = self._pinned.get(name, max(versions))
        template = versions[version].read_text(encoding="utf-8").strip()
        if not template:
            raise PromptError(f"prompt file is empty: {versions[version]}")
        return Prompt(name, version, template)


class ChainedPromptStore:
    """Looks in several stores in order, so a company can override a built-in prompt."""

    def __init__(self, stores: Sequence[PromptStore]) -> None:
        self._stores = list(stores)

    def get(self, name: str) -> Prompt:
        for store in self._stores:
            try:
                return store.get(name)
            except PromptNotFoundError:
                continue
        raise PromptNotFoundError(f"no prompt named '{name}' in any prompt store")


def builtin_store() -> FilePromptStore:
    """The generic prompt skeletons that ship with the core."""
    return FilePromptStore(Path(__file__).parent / "builtin")

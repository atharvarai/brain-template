"""The core must not contain any company's or industry's vocabulary.

Imports from ``examples`` are checked separately by import-linter (see pyproject.toml).
"""

import re

from tests.conftest import REPO_ROOT

CORE = REPO_ROOT / "core"

# Words that belong to the example domains. Extend this when a new pack is added.
DOMAIN_WORDS = [
    "skilvanta", "zipvaha", "edtech", "course", "courses", "learner", "learners",
    "mentor", "mentors", "enrollment", "placement", "curriculum",
    "rider", "riders", "payout", "payouts", "incentive",
]  # fmt: skip
PATTERN = re.compile(r"\b(" + "|".join(DOMAIN_WORDS) + r")\b", re.IGNORECASE)


def test_core_has_no_domain_vocabulary() -> None:
    offenders = []
    for path in sorted(CORE.rglob("*")):
        if not path.is_file() or path.suffix in {".pyc"}:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if match := PATTERN.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: '{match.group(0)}'")

    assert not offenders, "domain words found in core/:\n" + "\n".join(offenders)

"""Finds personal data and replaces it with placeholders such as ``[EMAIL]``.

Patterns plus checksums where the format has one (Luhn for cards, Verhoeff for Aadhaar), so
order ids and dates are not mistaken for card or ID numbers. Used before text is stored in
traces, logs and the audit log.
"""

import ipaddress
import re
from collections.abc import Callable, Iterable

from core.config import PiiKind

Validator = Callable[[str], bool]


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _luhn(text: str) -> bool:
    digits = [int(d) for d in _digits(text)]
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        checksum += digit
    return 13 <= len(digits) <= 19 and checksum % 10 == 0


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]  # fmt: skip
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]  # fmt: skip


def _verhoeff(text: str) -> bool:
    digits = _digits(text)
    if len(digits) != 12 or digits[0] in "01":
        return False
    check = 0
    for index, digit in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[index % 8][int(digit)]]
    return check == 0


def _phone(text: str) -> bool:
    return 10 <= len(_digits(text)) <= 13


def _ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False
    return True


# Order matters: longer, checksummed numbers are replaced before the looser phone pattern.
_DETECTORS: dict[PiiKind, tuple[re.Pattern[str], Validator | None]] = {
    PiiKind.CARD: (re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), _luhn),
    PiiKind.IN_AADHAAR: (re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"), _verhoeff),
    PiiKind.IN_PAN: (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None),
    PiiKind.EMAIL: (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"), None),
    PiiKind.IP_ADDRESS: (re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"), _ip),
    PiiKind.PHONE: (re.compile(r"(?<![\w+(])[+(]?\d[\d ().-]{8,16}\d(?!\w)"), _phone),
}


class PiiRedactor:
    def __init__(self, kinds: Iterable[PiiKind]) -> None:
        wanted = set(kinds)
        self._detectors = [(k, *d) for k, d in _DETECTORS.items() if k in wanted]

    def redact(self, text: str) -> str:
        for kind, pattern, valid in self._detectors:
            placeholder = f"[{kind.value.upper()}]"

            def replace(
                match: re.Match[str], valid: Validator | None = valid, mark: str = placeholder
            ) -> str:
                found = match.group(0)
                return mark if valid is None or valid(found) else found

            text = pattern.sub(replace, text)
        return text


class NoRedaction:
    def redact(self, text: str) -> str:
        return text

"""IPC (International Patent Classification) code extraction and hierarchical parsing.

IPC structure:  H   01   L   31/0725
                |   |    |   |   |
                |   |    |   |   subgroup
                |   |    |   group
                |   |    subclass
                |   class
                section
"""

from __future__ import annotations

import re
import unicodedata

from patiroha._types import IPCCode

# IPC section descriptions
IPC_SECTIONS: dict[str, str] = {
    "a": "生活必需品",
    "b": "処理操作; 運輸",
    "c": "化学; 冶金",
    "d": "繊維; 紙",
    "e": "固定構造物",
    "f": "機械工学; 照明; 加熱; 武器; 爆破",
    "g": "物理学",
    "h": "電気",
}


def parse_ipc(code: str) -> IPCCode:
    """Parse a single IPC code string into its hierarchical components.

    Args:
        code: An IPC code string, e.g. "H01L31/0725", "b32b27/00", "C08L".

    Returns:
        IPCCode dataclass with section, class_code, subclass, group, subgroup fields.

    Examples:
        >>> parse_ipc("H01L31/0725")
        IPCCode(raw='h01l31/0725', section='h', class_code='h01', subclass='h01l', group='31', subgroup='0725')
        >>> parse_ipc("B32B")
        IPCCode(raw='b32b', section='b', class_code='b32', subclass='b32b', group='', subgroup='')
    """
    code = unicodedata.normalize("NFKC", code).lower().strip()

    # Full IPC: e.g. "h01l31/0725"
    m = re.match(r"([a-z])(\d{2})([a-z])\s*(\d{1,4})/(\d{2,})", code)
    if m:
        sec, cls, sub, grp, sgrp = m.groups()
        return IPCCode(
            raw=code.replace(" ", ""),
            section=sec,
            class_code=f"{sec}{cls}",
            subclass=f"{sec}{cls}{sub}",
            group=grp,
            subgroup=sgrp,
        )

    # Group without subgroup: e.g. "h01l31"
    m = re.match(r"([a-z])(\d{2})([a-z])\s*(\d{1,4})$", code)
    if m:
        sec, cls, sub, grp = m.groups()
        return IPCCode(
            raw=code.replace(" ", ""),
            section=sec,
            class_code=f"{sec}{cls}",
            subclass=f"{sec}{cls}{sub}",
            group=grp,
            subgroup="",
        )

    # Subclass only: e.g. "h01l"
    m = re.match(r"([a-z])(\d{2})([a-z])$", code)
    if m:
        sec, cls, sub = m.groups()
        return IPCCode(
            raw=code,
            section=sec,
            class_code=f"{sec}{cls}",
            subclass=f"{sec}{cls}{sub}",
            group="",
            subgroup="",
        )

    # Fallback: just a section letter
    if len(code) == 1 and code.isalpha():
        return IPCCode(raw=code, section=code)

    return IPCCode(raw=code)


def extract_ipc(text: str, delimiter: str = ";") -> list[str]:
    """Extract IPC codes from a delimited text field as raw strings.

    Args:
        text: Raw IPC field text.
        delimiter: Separator character between IPC codes.

    Returns:
        List of normalized IPC code strings (lowercase, no spaces).
    """
    if not isinstance(text, str):
        return []

    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\(（][^)）]*[\)）]", " ", text)

    ipc_codes: list[str] = []
    parts = text.split(delimiter)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Full IPC code: e.g. "b32b 27/00"
        match = re.search(r"([a-z]\d{2}[a-z])\s*(\d{1,4}/\d{2,})", part)
        if match:
            ipc_code = match.group(1) + match.group(2)
            ipc_codes.append(ipc_code)
        else:
            # Main class only: e.g. "b32b"
            match_main = re.search(r"\b([a-z]\d{2}[a-z])\b", part)
            if match_main:
                ipc_codes.append(match_main.group(1))

    return ipc_codes


def extract_ipc_parsed(text: str, delimiter: str = ";") -> list[IPCCode]:
    """Extract and parse IPC codes into hierarchical IPCCode objects.

    Args:
        text: Raw IPC field text.
        delimiter: Separator character between IPC codes.

    Returns:
        List of parsed IPCCode dataclass instances.
    """
    raw_codes = extract_ipc(text, delimiter)
    return [parse_ipc(code) for code in raw_codes]

"""Parse a .sok level-collection file into individual .level files.

The .sok format (as produced by Sokoban YASC / sokobano.de collections) is:

    <header lines: Date of Last Change, Set:, Copyright:, Email:, Homepage:>
    <blank line>
    :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    <free-text comments/credits, blank-line separated>
    :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    <level id, e.g. "1a">
    <grid line>
    <grid line>
    ...
    Title: <title>
    Author: <author>
    <blank line>

    <level id>
    ...

i.e. records are separated by blank lines, and a record is a *level* record
(as opposed to a comment/header block) iff its last two lines start with
"Title:" and "Author:" respectively.

Usage:
    python parse_sok.py <path-to-.sok-file> [output-root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """Turn an arbitrary title string into a safe filename."""
    name = name.strip()
    name = name.replace(",", "")  # commas read awkwardly in filenames
    name = re.sub(r"[^\w\-. ]+", "_", name)  # drop anything not word/-/./space
    name = re.sub(r"\s+", " ", name).strip()
    return name or "untitled"


def read_text_any_encoding(path: Path) -> str:
    """.sok files are old collections and are frequently NOT utf-8
    (e.g. they use cp1252/latin-1 for the copyright symbol etc.)."""
    data = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    # last resort: replace undecodable bytes
    return data.decode("utf-8", errors="replace")


def parse_sok(path: str | Path) -> tuple[str, list[tuple[str, str, list[str]]]]:
    """Parse a .sok file.

    :return: (set_name, levels) where levels is a list of
        (level_id, title, grid_lines) tuples, in file order.
    """
    path = Path(path)
    raw = read_text_any_encoding(path)

    # Normalize line endings (handles \r\n, \r, \n) without touching
    # meaningful leading/trailing spaces within a line.
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # --- Pull the collection ("Set") name out of the header, if present ---
    set_name = path.stem
    for line in lines[:10]:
        m = re.match(r"^Set:\s*(.+)$", line)
        if m:
            set_name = m.group(1).strip()
            break

    # --- Split into blank-line-delimited records ---
    records: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip() == "":
            if current:
                records.append(current)
                current = []
        else:
            current.append(line)
    if current:
        records.append(current)

    levels: list[tuple[str, str, list[str]]] = []
    for rec in records:
        if len(rec) < 3:
            continue  # too short to be a level record (id + grid + title/author)

        # Locate the metadata block: it starts at the first "Title:" or
        # "Author:" line. Everything before that is the level id + grid.
        # Metadata line order/composition varies across the file (some
        # records have "Comment:"/"Comment-End:" or free-text notes after
        # Author, and at least one record has Author before Title), so we
        # scan for the two tags rather than assuming fixed positions.
        meta_start = None
        for i, line in enumerate(rec):
            if line.startswith("Title:") or line.startswith("Author:"):
                meta_start = i
                break
        if meta_start is None or meta_start < 2:
            continue  # a header / comment / credits block, not a level

        title = None
        for line in rec[meta_start:]:
            if line.startswith("Title:"):
                title = line[len("Title:"):].strip()
                break

        if title is None:
            continue  # no title found; not a proper level record

        level_id = rec[0].strip()
        grid_lines = rec[1:meta_start]

        levels.append((level_id, title, grid_lines))

    return set_name, levels


def write_levels(set_name: str, levels: list[tuple[str, str, list[str]]], output_root: Path) -> Path:
    game_dir = output_root / sanitize_filename(set_name)
    game_dir.mkdir(parents=True, exist_ok=True)

    used_names: dict[str, int] = {}
    for level_id, title, grid_lines in levels:
        base = sanitize_filename(title) if title else sanitize_filename(level_id)
        used_names[base] = used_names.get(base, 0) + 1
        fname = base if used_names[base] == 1 else f"{base} ({used_names[base]})"

        content = "\n".join(grid_lines) + "\n"
        (game_dir / f"{fname}.level").write_text(content, encoding="utf-8")

    return game_dir


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python parse_sok.py <path-to-.sok-file> [output-root]")
        sys.exit(1)

    sok_path = Path(sys.argv[1])
    output_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("levels")

    set_name, levels = parse_sok(sok_path)
    if not levels:
        print(f"No levels found in '{sok_path}'.")
        sys.exit(1)

    game_dir = write_levels(set_name, levels, output_root)
    print(f"Parsed {len(levels)} levels from '{set_name}' -> {game_dir}/")


if __name__ == "__main__":
    main()
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_TARGETS = [
    ROOT / "backend",
    ROOT / "scripts",
    ROOT / "README.md",
    ROOT / "backend" / "README_API.md",
]
ALLOWED_SUFFIXES = {".py", ".md", ".sh", ".bat", ".json"}
BANNED_SNIPPETS = [
    "from rag_engine import",
    "app.core.rag",
    "http://localhost:8000/v1/chat",
    "POST /v1/chat",
]


def _iter_source_files():
    for target in SEARCH_TARGETS:
        if target.is_file():
            yield target
            continue

        for path in target.rglob("*"):
            if path.is_file() and path.suffix in ALLOWED_SUFFIXES:
                if path.name == "test_cleanup_references.py":
                    continue
                yield path


def test_legacy_references_are_removed():
    violations = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in BANNED_SNIPPETS:
            if snippet in text:
                violations.append(f"{path.relative_to(ROOT)} contains '{snippet}'")

    assert not violations, "Legacy references found:\n" + "\n".join(violations)

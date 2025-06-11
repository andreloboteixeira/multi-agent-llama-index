import asyncio
import builtins
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from lobo.main import read_notes, write_refined_todos


@pytest.mark.asyncio
async def test_read_and_write_notes(tmp_path, monkeypatch):
    notes_file = tmp_path / "notes.md"
    refined_file = tmp_path / "refined_todos.md"

    notes_file.write_text("example notes")

    def fake_open(path, mode="r", *args, **kwargs):
        if path == "src/lobo/notes.md":
            return builtins.open(notes_file, mode, *args, **kwargs)
        if path == "src/lobo/refined_todos.md":
            return builtins.open(refined_file, mode, *args, **kwargs)
        return builtins.open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    content = await read_notes(None)
    assert content == "example notes"

    await write_refined_todos(None, "todo")
    assert refined_file.read_text() == "todo"

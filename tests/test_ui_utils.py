import types

import lightlygpt.utils.ui_utils as ui_utils


def test_format_agent_response():
    resp = ui_utils.format_agent_response("Hello")
    assert isinstance(resp, str)


def test_display_image_grid(monkeypatch):
    # Should not raise error with empty list
    class DummySt:
        def write(self, *a, **k):
            pass

        def columns(self, n):
            return [types.SimpleNamespace() for _ in range(n)]

    monkeypatch.setattr(ui_utils, "st", DummySt())
    ui_utils.display_image_grid([])

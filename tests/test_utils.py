import utils.clip_utils as clip_utils


def test_rgb_to_hex():
    assert clip_utils.rgb_to_hex((255, 255, 255)) == "#ffffff"
    assert clip_utils.rgb_to_hex((0, 0, 0)) == "#000000"
    assert clip_utils.rgb_to_hex((128, 64, 32)) == "#804020"


def test_check_dependency_present(monkeypatch):
    monkeypatch.setattr("builtins.__import__", lambda name, *a, **k: True)
    from utils.path_utils import check_dependency

    assert check_dependency("os") is True


def test_check_dependency_missing(monkeypatch):
    def raise_import(name, *a, **k):
        raise ImportError()

    monkeypatch.setattr("builtins.__import__", raise_import)
    from utils.path_utils import check_dependency

    assert check_dependency("nonexistentmodule") is False

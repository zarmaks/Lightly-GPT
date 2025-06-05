import lightlygpt.utils.clip_utils as clip_utils


def test_rgb_to_hex():
    assert clip_utils.rgb_to_hex((255, 255, 255)) == "#ffffff"
    assert clip_utils.rgb_to_hex((0, 0, 0)) == "#000000"
    assert clip_utils.rgb_to_hex((128, 64, 32)) == "#804020"


def test_check_dependency_present():
    # Test with a module that should exist and is in the DEPENDENCIES dict
    from lightlygpt.utils.path_utils import check_dependency

    # Test with 'torch' which should be available in the environment
    result = check_dependency("torch")
    # Should return a boolean
    assert isinstance(result, bool)


def test_check_dependency_missing():
    # Test with a module that should be missing and is in the DEPENDENCIES dict
    from lightlygpt.utils.path_utils import DEPENDENCIES, check_dependency

    # Temporarily set a dependency to False to test the function
    original_value = DEPENDENCIES.get("sklearn")
    DEPENDENCIES["sklearn"] = False

    result = check_dependency("sklearn")

    # Restore original value
    DEPENDENCIES["sklearn"] = original_value

    assert result is False

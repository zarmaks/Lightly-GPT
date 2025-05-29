import pytest
import types

import tools.analysis_tools as analysis_tools


class DummySessionState:
    def __init__(self):
        self.uploaded_images = []
        self.last_filtered_indices = []


@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    dummy_st = types.SimpleNamespace()
    dummy_st.session_state = DummySessionState()
    dummy_st.warning = lambda msg: None
    dummy_st.info = lambda msg: None
    dummy_st.success = lambda msg: None
    dummy_st.error = lambda msg: None
    dummy_st.write = lambda msg: None
    dummy_st.progress = lambda x: None
    monkeypatch.setattr(analysis_tools, "st", dummy_st)
    return dummy_st


def test_rgb_to_hex():
    assert analysis_tools.rgb_to_hex((255, 255, 255)) == "#ffffff"
    assert analysis_tools.rgb_to_hex((0, 0, 0)) == "#000000"
    assert analysis_tools.rgb_to_hex((128, 64, 32)) == "#804020"


def test_analyze_image_colors_no_images(patch_streamlit):
    patch_streamlit.session_state.uploaded_images = []
    patch_streamlit.session_state.last_filtered_indices = []
    result = analysis_tools.analyze_image_colors()
    assert "No images uploaded. Please upload images first." in result


def test_detect_bw_images_no_images(patch_streamlit):
    patch_streamlit.session_state.uploaded_images = []
    patch_streamlit.session_state.last_filtered_indices = []
    result = analysis_tools.detect_bw_images()
    assert "No images uploaded. Please upload images first." in result

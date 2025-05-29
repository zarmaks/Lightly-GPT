import pytest
import types

import tools.clip_tools as clip_tools


class DummyCollection:
    def query(self, query_embeddings, n_results):
        # Simulate a result with two images, one below and one above threshold
        return {"ids": [["img_0", "img_1"]], "distances": [[0.5, 0.8]]}


def dummy_generate_text_embedding(query):
    class Dummy:
        def tolist(self):
            return [0.1, 0.2, 0.3]

    return Dummy()


@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    dummy_st = types.SimpleNamespace()
    dummy_st.session_state = types.SimpleNamespace()
    dummy_st.session_state.processed = True
    dummy_st.session_state.collection = DummyCollection()
    dummy_st.session_state.uploaded_images = [
        types.SimpleNamespace(name="imgA"),
        types.SimpleNamespace(name="imgB"),
    ]
    dummy_st.session_state.last_filtered_indices = [0, 1]
    dummy_st.write = lambda msg: None
    dummy_st.error = lambda msg: None
    dummy_st.info = lambda msg: None
    dummy_st.success = lambda msg: None
    dummy_st.progress = lambda x: None
    monkeypatch.setattr(clip_tools, "st", dummy_st)
    monkeypatch.setattr(
        clip_tools, "generate_text_embedding", dummy_generate_text_embedding
    )
    monkeypatch.setattr(clip_tools, "display_image_grid", lambda *a, **k: None)
    monkeypatch.setattr(clip_tools, "get_active_indices", lambda: [0, 1])
    return dummy_st


def test_clip_image_search_tool_basic(patch_streamlit):
    result = clip_tools.clip_image_search_tool("cat", threshold=0.75)
    assert "I found 1 images matching" in result
    assert "imgA" in result


def test_clip_image_search_tool_no_processed(monkeypatch):
    import tools.clip_tools as clip_tools

    dummy_st = types.SimpleNamespace()
    dummy_st.session_state = types.SimpleNamespace()
    dummy_st.session_state.processed = False
    monkeypatch.setattr(clip_tools, "st", dummy_st)
    result = clip_tools.clip_image_search_tool("cat")
    assert "No processed images available" in result


def test_clip_image_search_tool_no_collection(monkeypatch):
    import tools.clip_tools as clip_tools

    dummy_st = types.SimpleNamespace()
    dummy_st.session_state = types.SimpleNamespace()
    dummy_st.session_state.processed = True
    dummy_st.session_state.collection = None
    monkeypatch.setattr(clip_tools, "st", dummy_st)
    result = clip_tools.clip_image_search_tool("cat")
    assert "No image collection available" in result


def test_clip_image_search_tool_no_images(monkeypatch):
    import tools.clip_tools as clip_tools

    dummy_st = types.SimpleNamespace()
    dummy_st.session_state = types.SimpleNamespace()
    dummy_st.session_state.processed = True
    dummy_st.session_state.collection = DummyCollection()
    dummy_st.session_state.uploaded_images = []
    monkeypatch.setattr(clip_tools, "st", dummy_st)
    result = clip_tools.clip_image_search_tool("cat")
    assert "No images uploaded" in result

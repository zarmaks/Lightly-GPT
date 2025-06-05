import types

import lightlygpt.utils.session_utils as session_utils


def test_initialize_session_state(monkeypatch):
    dummy_st = types.SimpleNamespace()
    dummy_st.session_state = {}
    monkeypatch.setattr(session_utils, "st", dummy_st)
    session_utils.initialize_session_state()
    assert "uploaded_images" in dummy_st.session_state
    assert "processed" in dummy_st.session_state
    assert "chroma_client" in dummy_st.session_state
    assert "collection" in dummy_st.session_state
    assert "image_metadata" in dummy_st.session_state
    assert "clip_model" in dummy_st.session_state
    assert "clip_processor" in dummy_st.session_state
    assert "messages" in dummy_st.session_state
    assert "agent" in dummy_st.session_state


def test_get_active_indices(monkeypatch):
    dummy_st = types.SimpleNamespace()
    dummy_st.session_state = types.SimpleNamespace()
    dummy_st.session_state.last_filtered_indices = [1, 2, 3]
    monkeypatch.setattr(session_utils, "st", dummy_st)
    indices = session_utils.get_active_indices()
    assert indices == [1, 2, 3]

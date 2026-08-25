from portrait_app.state import PortraitState


def test_descriptor_match_and_sample_rotation(tmp_path):
    state = PortraitState(tmp_path)
    person = state.create_person("Ana", "photo-1", [1.0, 0.0, 0.5])
    matched, score = state.match([0.99, 0.01, 0.49])
    assert matched["id"] == person["id"]
    assert score > 0.99
    for index in range(15):
        state.add_sample(person["id"], f"photo-{index + 2}", [1.0, 0.0, 0.5])
    saved = state.snapshot()["people"][0]
    assert len(saved["descriptors"]) == 12


def test_naming_a_gallery_face_reuses_its_saved_descriptor(tmp_path):
    state = PortraitState(tmp_path)
    state.remember_face("unknown-photo", [0.2, 0.4, 0.6])
    person = state.create_person("Leo", "unknown-photo")
    assert person["descriptors"] == [[0.2, 0.4, 0.6]]
    matched, _ = state.match([0.2, 0.4, 0.6])
    assert matched["name"] == "Leo"

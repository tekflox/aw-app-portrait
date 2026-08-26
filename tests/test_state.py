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


def test_unknown_collection_is_capped_and_adopted_when_named(tmp_path):
    state = PortraitState(tmp_path)
    descriptor = [1.0, 0.2, 0.4]
    first = state.register_capture("photo-0", descriptor)
    for index in range(1, 12):
        current = state.register_capture(f"photo-{index}", descriptor)
    assert current["collection_id"] == first["collection_id"]
    assert current["sample_count"] == 10

    person = state.create_person("Maya", "photo-4")
    assert person["photo_ids"] == [f"photo-{index}" for index in range(10)]
    assert state.snapshot()["unknown_clusters"] == []

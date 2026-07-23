from app.services.media import build_chunk_ranges


def test_chunks_cover_duration():
    chunks = build_chunk_ranges(1800, 300, 3)
    assert chunks[0] == (0.0, 300.0)
    assert chunks[-1][0] < 1800
    assert chunks[-1][0] + chunks[-1][1] == 1800


def test_one_short_chunk():
    assert build_chunk_ranges(50, 300, 3) == [(0.0, 50.0)]

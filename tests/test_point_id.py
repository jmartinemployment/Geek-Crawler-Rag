from geek_crawler_rag.qdrant_store import point_id


def test_point_id_deterministic():
    a = point_id("run-1", "page-1", 0)
    b = point_id("run-1", "page-1", 0)
    c = point_id("run-1", "page-1", 1)
    d = point_id("run-2", "page-1", 0)
    assert a == b
    assert a != c
    assert a != d

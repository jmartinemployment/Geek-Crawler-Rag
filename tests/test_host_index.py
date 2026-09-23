"""Host derivation for the index-exists check."""

from geek_crawler_rag.app import _host_candidates


def test_missing_scheme_is_tolerated():
    assert _host_candidates("pipedrive.com") == ["pipedrive.com", "www.pipedrive.com"]


def test_www_and_bare_are_both_tried():
    # Stored as distinct payload values, so a site indexed under www must still answer for the
    # bare domain the operator is more likely to type.
    assert _host_candidates("www.pipedrive.com") == ["www.pipedrive.com", "pipedrive.com"]


def test_a_bare_domain_finds_a_site_indexed_under_www():
    # The direction the comment above always described and the code never did. Which form a host is
    # indexed under follows the crawl seed, so typing the other one reported a fully indexed site
    # as missing -- https://www.medius.com indexed, https://medius.com not, one run, same moment.
    assert _host_candidates("medius.com") == ["medius.com", "www.medius.com"]


def test_the_typed_form_is_tried_first():
    # Most specific first: an exact match must not be beaten to the answer by the form we inferred.
    assert _host_candidates("www.medius.com")[0] == "www.medius.com"
    assert _host_candidates("medius.com")[0] == "medius.com"


def test_path_and_query_are_ignored():
    assert _host_candidates("https://x.com/pricing?plan=pro") == ["x.com", "www.x.com"]


def test_case_is_normalized():
    assert _host_candidates("HTTPS://WWW.Example.COM/x") == ["www.example.com", "example.com"]


def test_unusable_input_yields_no_host():
    # A URL that will not parse was never crawled, so no index can exist for it. Returning no host
    # IS the answer -- this is why no separate syntax check is needed alongside the index lookup.
    for raw in ["not a url", "asdfasdf", "", "   ", "http://", "localhost"]:
        assert _host_candidates(raw) == [], raw

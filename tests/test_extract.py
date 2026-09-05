from geek_crawler_rag.extract import extract_text_and_title, host_from_origin_or_url


def test_extract_title_and_body():
    html = """
    <html><head><title>  Partner Tool  </title></head>
    <body>
      <script>ignore()</script>
      <h1>Hello</h1>
      <p>Useful content about widgets.</p>
    </body></html>
    """
    text, title = extract_text_and_title(html)
    assert title == "Partner Tool"
    assert "Useful content about widgets" in text
    assert "ignore()" not in text


def test_extract_empty():
    assert extract_text_and_title(None) == ("", None)
    assert extract_text_and_title("") == ("", None)


def test_host_from_origin():
    assert host_from_origin_or_url("https://Example.COM", None) == "example.com"
    assert host_from_origin_or_url(None, "https://tools.acme.io/path") == "tools.acme.io"
    assert host_from_origin_or_url("example.com", None) == "example.com"

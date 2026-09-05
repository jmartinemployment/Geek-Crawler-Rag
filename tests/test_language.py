from geek_crawler_rag.language import detect_language, is_english


def test_english_detected():
    text = (
        "This partner integration guide explains how to connect your CRM "
        "with the marketing automation platform and sync contacts daily."
    )
    assert detect_language(text) == "en"
    assert is_english(text)


def test_spanish_not_english():
    text = (
        "Esta guía de integración explica cómo conectar su CRM con la plataforma "
        "de automatización de marketing y sincronizar contactos diariamente."
    )
    assert detect_language(text) == "es"
    assert not is_english(text)


def test_too_short_returns_none():
    assert detect_language("hi") is None
    assert not is_english("hi")

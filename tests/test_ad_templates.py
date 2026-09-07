from geek_crawler_rag.ad_templates import normalize_upsert_items, template_point_id
from geek_crawler_rag.models import AdTemplateUpsertItem


def test_template_point_id_stable():
    a = template_point_id("pas-linkedin")
    b = template_point_id("PAS-linkedin")
    assert a == b


def test_normalize_upsert_items_filters_short():
    items = normalize_upsert_items(
        [
            AdTemplateUpsertItem(id="a", name="A", body="short"),
            AdTemplateUpsertItem(id="b", name="B", body="Long enough template body here."),
            AdTemplateUpsertItem(id="b", name="B2", body="Duplicate id skipped with long body."),
        ]
    )
    assert len(items) == 1
    assert items[0].id == "b"

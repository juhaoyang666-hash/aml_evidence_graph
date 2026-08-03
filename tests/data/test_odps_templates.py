from pathlib import Path


def test_odps_templates_prohibit_cartesian_products() -> None:
    sql_root = Path(__file__).resolve().parents[2] / "sql" / "odps"
    templates = list(sql_root.glob("*.sql"))

    assert templates
    for template in templates:
        content = template.read_text(encoding="utf-8").lower()
        assert "cross join" not in content
        assert "left join" in content
        assert "\n  on " in content

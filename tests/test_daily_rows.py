import pytest

pytest.importorskip("earthkit.regrid")
pytest.importorskip("ecmwf.opendata")

from aifs_local.daily_verification import append_rows, existing_rows


def test_rows_roundtrip_and_dedupe_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert existing_rows() == set()

    rows = [dict(valid="2026-08-30T06:00:00", init="2026-08-29T06:00:00",
                 lead_h="24", param="2t", bias="0.1", mae="0.4", rmse="0.7")]
    append_rows(rows)
    append_rows([dict(rows[0], param="msl")])

    seen = existing_rows()
    assert ("2026-08-30T06:00:00", "24", "2t") in seen
    assert ("2026-08-30T06:00:00", "24", "msl") in seen
    assert len(seen) == 2

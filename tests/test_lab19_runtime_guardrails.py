import pandas as pd

from lab19_runtime import RunConfig, ensure_entity_audit_schema, valid_items_payload


def test_actual_runtime_config_applies_groq_free_tier_guardrails():
    smoke = RunConfig.for_mode("smoke")
    full = RunConfig.for_mode("full")

    assert smoke.extraction_max_chunks == 24
    assert smoke.groq_timeout_s >= 60.0
    assert smoke.groq_min_interval_s >= 20.0
    assert smoke.coref_batch_size == 4
    assert smoke.extraction_batch_size == 4

    assert full.golden_limit == 50
    assert 100 <= full.extraction_max_chunks <= 160
    assert full.groq_timeout_s >= 60.0
    assert full.groq_min_interval_s >= 20.0


def test_empty_entity_resolution_audit_has_stable_csv_schema():
    audit = ensure_entity_audit_schema(pd.DataFrame())
    assert list(audit.columns) == ["type", "left", "right", "similarity", "decision"]
    assert audit.empty


def test_items_payload_guard_rejects_semantically_malformed_json():
    assert valid_items_payload({"items": [{"chunk_id": "c1", "relations": []}]}) is True
    assert valid_items_payload({"items": ["not-an-object"]}) is False
    assert valid_items_payload({"items": "not-a-list"}) is False

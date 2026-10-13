import json

from xray.audit import AuditLog, verify


def test_chain_verifies_and_survives_reopen(tmp_path):
    p = tmp_path / "audit.jsonl"
    AuditLog(p).append("ai_recommendation", bag="A1", action="High Risk")
    AuditLog(p).append("officer_decision", bag="A1", decision="open_bag")  # reopened: continues the chain
    ok, _ = verify(p)
    assert ok


def test_edit_is_detected(tmp_path):
    p = tmp_path / "audit.jsonl"
    log = AuditLog(p)
    for i in range(3):
        log.append("officer_decision", bag=f"B{i}", decision="clear")
    lines = p.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    rec["data"]["decision"] = "open_bag"
    lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, msg = verify(p)
    assert not ok and "line 2" in msg


def test_deletion_is_detected(tmp_path):
    p = tmp_path / "audit.jsonl"
    log = AuditLog(p)
    for i in range(3):
        log.append("ai_recommendation", bag=f"C{i}", action="Safe")
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    ok, msg = verify(p)
    assert not ok and "chain broken" in msg

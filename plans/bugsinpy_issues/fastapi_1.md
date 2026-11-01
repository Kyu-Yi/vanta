# fastapi/1 — the project's own issue report

- issue: https://github.com/fastapi/fastapi/pull/1427
- resolved: via PR #1166
- fetched: 2026-10-25T12:31:18Z
- fix-commit-date: 2020-04-05T13:04:46Z
- comments-included: 0
- gate-tree-sha256: b0d60b808c2280eb9645f19cee355075152b45abdd13294bc6683170b03807fa
- gate-answer-sha256: deae38613f419c3411771f536dbbfb9741182c3f80d66546d904a22f5724d6a9
- gate-removals: 2
- raw-sha256: 600eff605825e5a6ef9eee7d9ad427c5c4c3a683c40ef34ba8da820aca0fdb7a
- content-sha256: dbe36e925533dab83c875b722b43d92b70767e28b63edf16aa520cd38c03863e
- raw-hints-sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- hints-sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

<!-- gate -->

{
  "files": [
    "tests/test_jsonable_encoder.py"
  ],
  "ok": true,
  "reason": null,
  "removals": [
    {
      "token": "response_model_exclude_none",
      "token_class": "ident",
      "reason": "token exists in the fix diff or the hidden test and nowhere in the tree the Creator may read",
      "occurrences_in_answer": 71,
      "occurrences_in_vanta_tree": 0,
      "action": "stripped"
    },
    {
      "token": "response_model_exclude_defaults",
      "token_class": "ident",
      "reason": "token exists in the fix diff or the hidden test and nowhere in the tree the Creator may read",
      "occurrences_in_answer": 70,
      "occurrences_in_vanta_tree": 0,
      "action": "stripped"
    }
  ],
  "ungated_prose": [],
  "hints": null
}

<!-- raw -->

# Documented additional parameters

These are included in a recent PR (https://github.com/tiangolo/fastapi/pull/1166) but not in the docs yet.
* response_model_exclude_none
* response_model_exclude_defaults

<!-- content -->

# Documented additional parameters

These are included in a recent PR (https://github.com/tiangolo/fastapi/pull/1166) but not in the docs yet.
* [redacted]
* [redacted]

<!-- raw-hints -->


<!-- hints -->



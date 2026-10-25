# PySnooper/3 — the project's own issue report

- issue: https://github.com/cool-RR/PySnooper/issues/2
- resolved: direct
- fetched: 2026-10-25T12:31:38Z
- fix-commit-date: 2019-04-22T12:10:34Z
- comments-included: 0
- gate-tree-sha256: fb46c3f1f48f1c939f8b161a3a8c8a58ef4c8eaa8ea993f3fd2af07ee5ac7064
- gate-answer-sha256: a672a23849b8e61cba39c85cdc6f60dde60dd8bfab4cb56824ff13d684ec6d50
- gate-removals: 0
- raw-sha256: 183f103970efc4bca38265c22fd8062161351898de509988a70eeeacfce38e24
- content-sha256: 183f103970efc4bca38265c22fd8062161351898de509988a70eeeacfce38e24
- raw-hints-sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- hints-sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

<!-- gate -->

{
  "files": [
    "tests/test_pysnooper.py"
  ],
  "ok": true,
  "reason": null,
  "removals": [],
  "ungated_prose": [],
  "hints": null
}

<!-- raw -->

# NameError: global name 'output_path' is not defined

https://github.com/cool-RR/PySnooper/blob/6e3d797be3fa0a746fb5b1b7c7fea78eb926c208/pysnooper/pysnooper.py#L26

should probably read
```
with open(output, 'a') as output_file:
```

<!-- content -->

# NameError: global name 'output_path' is not defined

https://github.com/cool-RR/PySnooper/blob/6e3d797be3fa0a746fb5b1b7c7fea78eb926c208/pysnooper/pysnooper.py#L26

should probably read
```
with open(output, 'a') as output_file:
```

<!-- raw-hints -->


<!-- hints -->



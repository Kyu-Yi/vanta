# fastapi/2 — the project's own issue report

- issue: https://github.com/fastapi/fastapi/issues/998
- resolved: via PR #1122
- fetched: 2026-10-28T06:57:06Z
- fix-commit-date: 2020-03-30T18:45:05Z
- comments-included: 1
- gate-tree-sha256: ee7df3c770b8c003569ff8c54972ca643df29169e58f1e6520a9d11486091ecf
- gate-answer-sha256: 0b77d80e601e9380ca7cb0be39aa304907ad8b717b7e050793bb1c51d9e2f8da
- gate-removals: 0
- raw-sha256: 8fb75e1ce259eb68babd6dd553d3ce0f71a9931f998bda2b546a25cc54b742b2
- content-sha256: 8fb75e1ce259eb68babd6dd553d3ce0f71a9931f998bda2b546a25cc54b742b2
- raw-hints-sha256: 7b3caaa805e993bc0bb45d18181d5f0ded2ebb7da2a8befb5b2e9810d71950d7
- hints-sha256: 7b3caaa805e993bc0bb45d18181d5f0ded2ebb7da2a8befb5b2e9810d71950d7

<!-- gate -->

{
  "files": [
    "tests/test_ws_router.py"
  ],
  "ok": true,
  "reason": null,
  "removals": [],
  "ungated_prose": [],
  "hints": {
    "removals": [],
    "ungated_prose": []
  }
}

<!-- raw -->

# Dependency override websocket broken

### Describe the bug

Dependency override does not work for websockets. 

The function `add_api_websocket_route` does not add `dependency_overrides_provider` to `APIWebSocketRoute`.

### To Reproduce

Create a simple app with websockets and test it with override.

### Expected behavior

The overrides should be taken into account, but the test uses the original dependency. 

### Environment

- OS: Windows
- FastAPI version: 0.49.0
- Python version: 3.6.8

<!-- content -->

# Dependency override websocket broken

### Describe the bug

Dependency override does not work for websockets. 

The function `add_api_websocket_route` does not add `dependency_overrides_provider` to `APIWebSocketRoute`.

### To Reproduce

Create a simple app with websockets and test it with override.

### Expected behavior

The overrides should be taken into account, but the test uses the original dependency. 

### Environment

- OS: Windows
- FastAPI version: 0.49.0
- Python version: 3.6.8

<!-- raw-hints -->

## Comments

### amitlissack — 2020-03-16T16:23:21Z

`APIRouter.add_api_websocket_route` should be modified to construct the `APIWebSocketRoute` like this:
```
route = APIWebSocketRoute(
            path,
            endpoint=endpoint,
            name=name,
            dependency_overrides_provider=self.dependency_overrides_provider,
        )
```

<!-- hints -->

## Comments

### amitlissack — 2020-03-16T16:23:21Z

`APIRouter.add_api_websocket_route` should be modified to construct the `APIWebSocketRoute` like this:
```
route = APIWebSocketRoute(
            path,
            endpoint=endpoint,
            name=name,
            dependency_overrides_provider=self.dependency_overrides_provider,
        )
```


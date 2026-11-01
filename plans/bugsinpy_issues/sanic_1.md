# sanic/1 — the project's own issue report

- issue: https://github.com/sanic-org/sanic/issues/1847
- resolved: direct
- fetched: 2026-10-25T12:31:38Z
- fix-commit-date: 2020-05-13T23:54:47Z
- comments-included: 0
- gate-tree-sha256: 486349791fc83ba657cd6127b52a4f147947bbb432f12146cbf26ca643040bea
- gate-answer-sha256: 130363f560fd2d8158fbc82ddb7a9d4be70a0420587df3277e3027aa91c4d777
- gate-removals: 0
- raw-sha256: 4163b000081e010bc0873f29911fd8f88a5cefaf0440a696fd2fe51834a45cab
- content-sha256: 4163b000081e010bc0873f29911fd8f88a5cefaf0440a696fd2fe51834a45cab
- raw-hints-sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- hints-sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

<!-- gate -->

{
  "files": [
    "tests/test_blueprints.py"
  ],
  "ok": true,
  "reason": null,
  "removals": [],
  "ungated_prose": [],
  "hints": null
}

<!-- raw -->

# "Named Response Middleware" executed in wrong order

**Describe the bug**
PR https://github.com/huge-success/sanic/pull/1690 Introduced "named response middlware" that is, middleware which is only executed in a given request context. For example a blueprint middleware is only executed on a route which is defined in _that_ blueprint.
There was a copy+paste error in the `register_named_middleware` function, here: https://github.com/huge-success/sanic/blob/e7001b00747b659f7042b0534802b936ee8a53e0/sanic/app.py#L656
When registering a "response" middleware, they are supposed to be added to the left in reverse. So `appendleft()` should be used instead of  `append()`. The correct behavior is seen in the normal `register_middleware` function.

**Code snippet**
See these two examples, the first using normal middleware, and the second using named middleware:
```
from sanic import Sanic
from sanic.response import text
app = Sanic(__name__)

@app.middleware('request')
def request_middleware_1(request):
    print('1')

@app.middleware('request')
def request_middleware_2(request):
    print('2')

@app.middleware('request')
def request_middleware_3(request):
    print('3')

@app.middleware('response')
def resp_middleware_4(request, response):
    print('4')

@app.middleware('response')
def resp_middleware_5(request, response):
    print('5')

@app.middleware('response')
def resp_middleware_6(request, response):
    print('6')

@app.route('/')
def pop_handler(request):
    return text('hello world')

app.run(host="0.0.0.0", port=8000, debug=True, auto_reload=False)
```
vs:
```
from sanic import Sanic, Blueprint
from sanic.response import text

app = Sanic(__name__)
bp = Blueprint("bp_"+__name__)

@bp.middleware('request')
def request_middleware_1(request):
    print('1')

@bp.middleware('request')
def request_middleware_2(request):
    print('2')

@bp.middleware('request')
def request_middleware_3(request):
    print('3')

@bp.middleware('response')
def resp_middleware_4(request, response):
    print('4')

@bp.middleware('response')
def resp_middleware_5(request, response):
    print('5')

@bp.middleware('response')
def resp_middleware_6(request, response):
    print('6')

@bp.route('/')
def pop_handler(request):
    return text('hello world')

app.blueprint(bp, url_prefix='/bp')

app.run(host="0.0.0.0", port=8000, debug=True, auto_reload=False)
```

**Expected behavior**
See the first snippet prints "1 2 3 6 5 4" (correct) but the second snippet prints "1 2 3 4 5 6". This should match the first.

**Additional Context**
This bug is _similar to_ but not the same as https://github.com/huge-success/sanic/issues/1845
This bug was uncovered while looking deeper into https://github.com/huge-success/sanic/issues/1845

<!-- content -->

# "Named Response Middleware" executed in wrong order

**Describe the bug**
PR https://github.com/huge-success/sanic/pull/1690 Introduced "named response middlware" that is, middleware which is only executed in a given request context. For example a blueprint middleware is only executed on a route which is defined in _that_ blueprint.
There was a copy+paste error in the `register_named_middleware` function, here: https://github.com/huge-success/sanic/blob/e7001b00747b659f7042b0534802b936ee8a53e0/sanic/app.py#L656
When registering a "response" middleware, they are supposed to be added to the left in reverse. So `appendleft()` should be used instead of  `append()`. The correct behavior is seen in the normal `register_middleware` function.

**Code snippet**
See these two examples, the first using normal middleware, and the second using named middleware:
```
from sanic import Sanic
from sanic.response import text
app = Sanic(__name__)

@app.middleware('request')
def request_middleware_1(request):
    print('1')

@app.middleware('request')
def request_middleware_2(request):
    print('2')

@app.middleware('request')
def request_middleware_3(request):
    print('3')

@app.middleware('response')
def resp_middleware_4(request, response):
    print('4')

@app.middleware('response')
def resp_middleware_5(request, response):
    print('5')

@app.middleware('response')
def resp_middleware_6(request, response):
    print('6')

@app.route('/')
def pop_handler(request):
    return text('hello world')

app.run(host="0.0.0.0", port=8000, debug=True, auto_reload=False)
```
vs:
```
from sanic import Sanic, Blueprint
from sanic.response import text

app = Sanic(__name__)
bp = Blueprint("bp_"+__name__)

@bp.middleware('request')
def request_middleware_1(request):
    print('1')

@bp.middleware('request')
def request_middleware_2(request):
    print('2')

@bp.middleware('request')
def request_middleware_3(request):
    print('3')

@bp.middleware('response')
def resp_middleware_4(request, response):
    print('4')

@bp.middleware('response')
def resp_middleware_5(request, response):
    print('5')

@bp.middleware('response')
def resp_middleware_6(request, response):
    print('6')

@bp.route('/')
def pop_handler(request):
    return text('hello world')

app.blueprint(bp, url_prefix='/bp')

app.run(host="0.0.0.0", port=8000, debug=True, auto_reload=False)
```

**Expected behavior**
See the first snippet prints "1 2 3 6 5 4" (correct) but the second snippet prints "1 2 3 4 5 6". This should match the first.

**Additional Context**
This bug is _similar to_ but not the same as https://github.com/huge-success/sanic/issues/1845
This bug was uncovered while looking deeper into https://github.com/huge-success/sanic/issues/1845

<!-- raw-hints -->


<!-- hints -->



# Testing patterns

Code-quality rubric entries for **testing patterns** (10 entries). See `README.md` for the full topic index.

## Assert on the exception message with pytest.raises(match=), not bare pytest.raises  ·  `high`

When testing for an exception, pass match= to pytest.raises so the test pins the specific error, not just the type. Remember match= is an re.search (partial, regex) match: escape regex metacharacters and anchor when you mean an exact message.


**Why:** A bare pytest.raises(ValueError) passes for ANY ValueError, including one accidentally raised by a typo in your own test setup before the real code runs, so the test gives false confidence. And because match= uses re.search, a message like 'rate (1.5)' silently treats the parentheses as a regex group and 'a.b' matches 'axb', so an unescaped pattern can pass against the wrong message.


**Avoid:**
```python
import pytest

def test_rejects_negative_dim():
    with pytest.raises(ValueError):  # any ValueError passes, even an unrelated one
        build_layer(dim=-4)
```

**Prefer:**
```python
import re
import pytest

def test_rejects_negative_dim():
    with pytest.raises(ValueError, match=re.escape("dim must be positive (got -4)")):
        build_layer(dim=-4)
```

Source: [pytest documentation - How to write and report assertions](https://docs.pytest.org/en/stable/how-to/assert.html)

## Patch where the name is looked up, not where it is defined  ·  `high`

When mocking with monkeypatch.setattr or unittest.mock.patch, target the name in the module that USES it, not the module that defines it. A `from x import y` in the system-under-test binds a local name that patching the origin module will not affect.


**Why:** Because `from client import fetch` copies the reference into the SUT's namespace, patching 'client.fetch' leaves the SUT calling the real function. The test then exercises live code (network, DB, mE5 encoder) while appearing to pass with a mock, the worst kind of false confidence because it looks green.


**Avoid:**
```python
# service.py:  from client import fetch ;  def run(): return fetch(url)
mocker.patch("client.fetch", return_value=42)  # SUT still calls the real fetch
assert service.run() == 42
```

**Prefer:**
```python
# service.py:  from client import fetch ;  def run(): return fetch(url)
mocker.patch("service.fetch", return_value=42)  # patch where it is looked up
assert service.run() == 42
```

Source: [Python docs - unittest.mock (Where to patch)](https://docs.python.org/3/library/unittest.mock.html)

## Spec your mocks (autospec) so they fail when the real API changes  ·  `high`

Create mocks with create_autospec / autospec=True (or spec=) instead of a bare Mock/MagicMock. A specced mock validates attribute names and call signatures against the real object.


**Why:** A bare MagicMock answers to any attribute and accepts any arguments, so when the real function is renamed or its signature changes, the test keeps passing against an API that no longer exists, while production breaks. An autospec'd mock raises TypeError/AttributeError on the same misuse the real object would, catching drift.


**Avoid:**
```python
from unittest.mock import MagicMock

encoder = MagicMock()
encoder.encdoe(text)        # typo: silently returns a Mock, test passes
encoder.encode(text, foo=1) # wrong signature: also passes
```

**Prefer:**
```python
from unittest.mock import create_autospec
from minicoil_v2.encoder import MiniCoilEncoder

encoder = create_autospec(MiniCoilEncoder, instance=True)
encoder.encode(text)        # AttributeError on a typo, TypeError on bad args
```

Source: [Python docs - unittest.mock (Autospeccing)](https://docs.python.org/3/library/unittest.mock.html)

## Verify observable results, not that a mock was called  ·  `high`

Assert on the value/state the code produces, not merely that a collaborator method was invoked. A test whose only assertion is mock.assert_called_once() restates the implementation and breaks on harmless refactors while missing real bugs.


**Why:** Mockist (behavior) verification couples the test to HOW the code works rather than WHAT it produces; Fowler notes such tests are 'more coupled to the implementation' and that 'changing the nature of calls to collaborators usually cause a mockist test to break'. The test passes as long as the call happens, even if the returned/stored result is wrong.


**Avoid:**
```python
def test_store_writes(mock_store):
    save_vectors(vecs, store=mock_store)
    mock_store.upsert.assert_called_once()  # tests the mock, not the outcome
```

**Prefer:**
```python
def test_store_writes():
    store = InMemoryStore()            # a fake with real behavior
    save_vectors(vecs, store=store)
    assert store.count() == len(vecs)  # assert the observable result
```

Source: [Martin Fowler - Mocks Aren't Stubs](https://martinfowler.com/articles/mocksArentStubs.html)

## Don't mock what you don't own; wrap third-party APIs and inject them  ·  `medium`

Don't patch the internals of third-party libraries (HTTP clients, the Qdrant client, transformers). Wrap them in a thin façade you own, inject that façade as a dependency, and substitute a fake/specced double in tests.


**Why:** Mocking a library you don't control means 'the purpose of the test is drowning in boilerplate necessary to mimic the API of an HTTP client that can change at any time' (Schlawack). The mock encodes your assumptions about the library; when the library changes, the mock lies and tests stay green. A façade gives one small seam to fake and improves the production design.


**Avoid:**
```python
def test_search(monkeypatch):
    # three layers of mocks mimicking the raw qdrant client internals
    monkeypatch.setattr("qdrant_client.QdrantClient.search", lambda *a, **k: [...])
```

**Prefer:**
```python
class VectorStore:                       # façade you own
    def __init__(self, client): self._c = client
    def search(self, q): return self._c.search(...)

def test_search():
    svc = SearchService(store=FakeStore(hits=[...]))  # inject a fake
    assert svc.top(q) == [...]
```

Source: [Hynek Schlawack - Don't Mock What You Don't Own in 5 Minutes](https://hynek.me/articles/what-to-mock-in-5-mins/)

## Return a factory function from a fixture when a test needs multiple/parameterized instances  ·  `medium`

When a test needs several configured objects, or the same object built with different arguments, have the fixture return a function (a factory) rather than a single fixed object. Let the factory register cleanup for everything it creates.


**Why:** A fixture that returns one pre-built object forces tests to mutate shared state or duplicate setup when they need a second variant. The factory-as-fixture pattern lets one fixture serve many shapes 'called multiple times in the test' and centralize teardown for all instances it produced, keeping construction logic out of the test body.


**Avoid:**
```python
@pytest.fixture
def concept():
    return Concept(id=1, en="cat", es="gato")  # only one shape; tests needing two are stuck

def test_pair(concept): ...
```

**Prefer:**
```python
@pytest.fixture
def make_concept():
    def _make(en, es, id=0):
        return Concept(id=id, en=en, es=es)
    return _make

def test_pair(make_concept):
    a, b = make_concept("cat", "gato"), make_concept("dog", "perro")
```

Source: [pytest documentation - How to use fixtures (Factories as fixtures)](https://docs.pytest.org/en/stable/how-to/fixtures.html)

## Use property-based tests (Hypothesis) for pure logic and invariants  ·  `medium`

For pure functions and algorithmic logic (tokenization, ID renumbering, vocab pruning, normalization), express the property that must always hold and let Hypothesis @given generate inputs, instead of hand-picking a few examples.


**Why:** Hand-written examples test only the cases you thought of and miss the boundary inputs that break the code. Hypothesis generates ~100 inputs per run and, on failure, shrinks to a minimal falsifying example (e.g. 'Falsifying example: test_integers(n=50,)'), pointing straight at the smallest reproducer and growing a regression corpus over time.


**Avoid:**
```python
def test_roundtrip():
    assert decode(encode("cat")) == "cat"
    assert decode(encode("gato")) == "gato"  # only the inputs you imagined
```

**Prefer:**
```python
from hypothesis import given, strategies as st

@given(st.text())
def test_roundtrip(s):
    assert decode(encode(s)) == s  # invariant must hold for all inputs
```

Source: [Hypothesis documentation - Quickstart](https://hypothesis.readthedocs.io/en/latest/quickstart.html)

## Keep control flow and computed expectations out of test bodies  ·  `medium`

No loops, if/else, or try/except wrapping assertions, and no recomputing the expected value with the same logic the code uses. Drive variation with @pytest.mark.parametrize (each case fails independently and is named in the report) and write expected values as literals.


**Why:** A for-loop over assertions stops at the first failure and hides every other broken case in one opaque error; an if/else in a test can mirror the implementation's bug so the test passes against broken code; and computing the expected value the same way the SUT does means the test asserts the code equals itself. Parametrize reports each case separately and forces hard-coded expectations.


**Avoid:**
```python
def test_scores():
    for q, doc, exp in CASES:
        s = score(q, doc)
        if exp > 0:                 # logic mirrors the impl
            assert s == q.idf * doc.tf  # recomputed, not a literal
```

**Prefer:**
```python
@pytest.mark.parametrize("q, doc, expected", [
    ("cat", "a cat sat", 1.4),
    ("cat", "no match", 0.0),
])
def test_scores(q, doc, expected):
    assert score(q, doc) == pytest.approx(expected)
```

Source: [pytest documentation - How to parametrize fixtures and test functions](https://docs.pytest.org/en/stable/how-to/parametrize.html)

## Use tmp_path and monkeypatch for filesystem and env, never manual setup/teardown  ·  `medium`

For files use the tmp_path fixture (a unique pathlib.Path per test), not hand-rolled tempfile dirs or repo-relative paths; the legacy tmpdir (py.path.local) is superseded. For env vars and attributes use monkeypatch.setenv/setattr, never os.environ assignment with a finally block.


**Why:** Manual os.environ edits or tempdir creation leak state into other tests when an assertion fails before the teardown line runs, producing order-dependent flakiness. monkeypatch guarantees 'all modifications will be undone after the requesting test function or fixture has finished', and tmp_path gives each test an isolated directory automatically, removing the cleanup code entirely.


**Avoid:**
```python
def test_loads_config():
    os.environ["MINICOIL_DEVICE"] = "cpu"   # leaks if the assert below raises
    assert load_settings().device == "cpu"
    del os.environ["MINICOIL_DEVICE"]
```

**Prefer:**
```python
def test_loads_config(monkeypatch, tmp_path):
    monkeypatch.setenv("MINICOIL_DEVICE", "cpu")  # auto-undone at teardown
    (tmp_path / "cfg.toml").write_text("...")
    assert load_settings().device == "cpu"
```

Source: [pytest documentation - How to monkeypatch/mock modules and environments](https://docs.pytest.org/en/stable/how-to/monkeypatch.html)

## Register custom markers and run with --strict-markers  ·  `low`

Declare every custom marker (slow, integration, gpu) in pyproject.toml's [tool.pytest.ini_options] markers list and enable --strict-markers via addopts, so a mistyped marker is a collection error rather than a silently ignored decorator.


**Why:** An unregistered marker only emits a PytestUnknownMarkWarning, and a typo like @pytest.mark.itegration silently does nothing, so the test you meant to gate as integration runs everywhere (or is never deselected). --strict-markers turns mistyped names into build-breaking errors at collection time and documents the suite's marker vocabulary.


**Avoid:**
```python
# no markers registered, no strict mode
@pytest.mark.itegration   # typo: silently ignored, test is never deselected
def test_hits_qdrant(): ...
```

**Prefer:**
```python
# pyproject.toml
# [tool.pytest.ini_options]
# addopts = "--strict-markers"
# markers = ["integration: needs a live Qdrant", "slow: > 1s"]
@pytest.mark.integration
def test_hits_qdrant(): ...
```

Source: [pytest documentation - How to mark test functions with attributes](https://docs.pytest.org/en/stable/how-to/mark.html)

How to read a failing CI log: find the first real error, skip the noise.

```
Run pip install -e .[test]
  DEPRECATION: setup.py develop is deprecated            <- noise
...
tests/test_cache.py::test_ttl_expiry PASSED
tests/test_cache.py::test_lru_evicts_oldest FAILED       <- first failure
tests/test_cache.py::test_stats ERROR                    <- may follow from it
tests/test_cli.py::test_clear FAILED                     <- may follow from it
=================================== FAILURES ===================================
____________________________ test_lru_evicts_oldest ____________________________
    def test_lru_evicts_oldest():
        cache = LRUCache(maxsize=2)
        cache["a"] = 1; cache["b"] = 2; cache["a"]; cache["c"] = 3
>       assert "b" not in cache
E       AssertionError: assert 'b' not in LRUCache({'a': 1, 'b': 2, 'c': 3})
src/cache/lru.py:41: in __setitem__                      <- innermost project frame
    if len(self._data) > self.maxsize + 1:
```

What to take from it:

- Failing test: `tests/test_cache.py::test_lru_evicts_oldest`.
- Error: the cache holds 3 items with `maxsize=2`, so nothing was evicted.
- Where: `src/cache/lru.py:41`. The eviction check compares against
  `maxsize + 1`, an off-by-one.
- Reproduce with: `pytest tests/test_cache.py::test_lru_evicts_oldest`.
- Then run the full suite. `test_stats` and `test_clear` probably fail for the
  same cause, so check them again after the fix before investigating them
  separately.

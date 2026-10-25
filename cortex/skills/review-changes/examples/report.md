Review of `git diff main...HEAD` (3 files, +84 -12): the new `/files/<name>`
download route and the pagination rewrite.

**1. Critical: path traversal in the file download**: `app/routes.py:57`
`send_file(UPLOAD_DIR / name)` does not check that the result stays inside
`UPLOAD_DIR`, so a crafted name reads any file the server can read.

```
$ PYTHONPATH=. python /tmp/cortex-review.k2Xa91/repro_traversal.py
GET /files/..%2F..%2Fetc%2Fpasswd -> 200, body starts: root:x:0:0:root
```
Fix direction: resolve the path and reject it unless it is inside
`UPLOAD_DIR`.

**2. High: the last partial page is dropped**: `app/paging.py:23`
`pages = total // size` drops the remainder handling the old code had, so when
`total % size != 0` the final page is never requested.

```
$ PYTHONPATH=. python /tmp/cortex-review.k2Xa91/repro_paging.py
35 items, size 10: got 3 pages, expected 4 (items 31..35 missing)
```
Fix direction: `pages = -(-total // size)`.

**Checked and fine**: auth on the new route (the decorator runs, and an
anonymous request gets 401); unicode file names; an empty upload directory;
the callers of `paginate` (2, both pass positive sizes).

A good commit message:

```
Keep the last page when the total is not a multiple of the page size

paginate() computed the page count as total // size, so a list of 35
items in pages of 10 was served as 3 pages and items 31-35 were never
returned. The count now rounds up, and a test covers a partial last
page, an exact multiple and an empty list.
```

Why it works: the subject says what changed in the imperative and fits in 72
characters. The body says what was wrong, what changed and how it is tested.

Weak messages, for contrast:

- `fix bug`: says nothing.
- `Updated paging.py and test_paging.py`: lists files instead of the change.
- `Fixed the pagination issue that was causing problems with some lists.`:
  vague, past tense, and ends with a period.

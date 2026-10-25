Request: "How do we export reports, and how would I add a --json flag?"

**Summary**: `report export` is a Click command that loads rows through
`ReportStore.fetch` and writes them with the CSV writer in `report/formats.py`.
The format is fixed to CSV; adding JSON means a new writer and a flag that picks
between them.

**Map**:
1. `app/cli.py:88`: `export_cmd`, the `report export` command; parses
   `--since` and `--out`, then calls `run_export`.
2. `app/report/export.py:14`: `run_export`, which loads rows and opens the
   output file.
3. `app/report/store.py:52`: `ReportStore.fetch`, which queries rows newer than
   `since` and returns a list of `Row`.
4. `app/report/formats.py:9`: `write_csv`, which writes the header from
   `Row._fields` and one line per row.

**Plan**:
1. `app/report/formats.py:9`: add `write_json(rows, out)` next to `write_csv`,
   writing a list of `row._asdict()`. Keep `write_csv` unchanged.
2. `app/report/export.py:14`: give `run_export` a `fmt="csv"` parameter and pick
   the writer from a dict `{"csv": write_csv, "json": write_json}`. The default
   keeps the two other callers (`app/jobs/nightly.py:31`,
   `tests/test_export.py:12`) working.
3. `app/cli.py:88`: add `--json` as a flag that passes `fmt="json"`. Copy how
   `--compact` is declared on `app/cli.py:61`.
4. Tests: in `tests/test_export.py`, add a case that exports two rows with
   `fmt="json"` and checks the parsed output. Run `pytest tests/test_export.py`,
   then `pytest`.

Risks: `Row` holds `datetime` values (`app/report/store.py:12`), which
`json.dump` cannot write. Step 1 needs `default=str` or explicit conversion.

**Open questions**: should `--out` default to `.json` when `--json` is given?
The CLI has no precedent for this.

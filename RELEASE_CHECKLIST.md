# RELEASE CHECKLIST — before making this repo public

This repo is **private during double-blind review**. A reproduction README and
public code inherently de-anonymize the work, which is correct for a *post-
acceptance* release but must NOT be exposed while the BIBM submission is under
review. Work through this checklist before flipping the repo to public (i.e. at
camera-ready, after acceptance).

Do not link this repo from the submitted paper. During review, the paper's
"we release code at…" line should point to an anonymized mirror
(e.g. https://anonymous.4open.science), not to this named repository.

---

## 1. Anonymization scrub (BLOCKING — must all be clean before public)

Run this and resolve every hit:

```powershell
Select-String -Path *.py, *.md, prompts\*, data\*, results\*.json -Pattern "AI-READI|AI_READI|hader|PycharmProjects|C:\\Users"
```

Known items to fix:

- [ ] **`run_benchmark_v4.py` (~line 324):** console banner prints
      `"AI-READI SSSOM BENCHMARK v4 …"`. Replace `AI-READI` with a neutral
      label (e.g. `"SSSOM BENCHMARK v4"`) — the dataset name is suppressed in
      the paper and must not appear in released code.
- [ ] **Any `C:\Users\hader\...` absolute paths** in docstrings, comments, or
      the `config.csv` field inside results JSONs. The results files record the
      CSV path used at run time; check `benchmark_results_v4.json` and
      `holdout_results.json` for embedded Windows paths and rewrite them to the
      relative `data/…` form.
- [ ] **`RUNBOOK.md`** — scan for institution names, the curator's name, or
      absolute paths.
- [ ] Confirm no OpenRouter API key is committed anywhere
      (`Select-String -Path *.py -Pattern "sk-or-"`).

## 2. Curator spreadsheet (BLOCKING)

- [ ] Open `data/curator_spreadsheet.csv`. Confirm no column contains the
      curator's real name, initials, email, or any annotator identifier. If an
      annotator column exists, replace values with a neutral tag
      (e.g. `curator_A`).
- [ ] Confirm no source term or target concept text constitutes PHI. (Per the
      paper, the graded labels are not PHI and the mappings are public — but
      verify against the originating dataset's redistribution terms.)

## 3. Dataset name in the paper vs. repo

- [ ] The paper suppresses the primary dataset's identity for double-blind
      review. Decide whether the camera-ready paper de-suppresses it. Whatever
      you choose, the repo and the camera-ready paper must be consistent — do
      not name the dataset in one and hide it in the other.

## 4. Fill the README TODOs

- [ ] BibTeX `author` field and citation key
- [ ] Contact / issue-tracker line
- [ ] Update the citation once the paper has a DOI or proceedings page

## 5. License

- [ ] Add a `LICENSE` file (the README states MIT for code). Confirm the data
      CSVs can be redistributed under that license, or add a separate data
      license note if the OHDSI Athena / primary-dataset terms require it.

## 6. Repo hygiene

- [ ] Confirm `.gitignore` excludes `.idea/`, `__pycache__/`, `.venv/`, and
      LaTeX build artifacts, and that none were committed
      (`git ls-files | Select-String "\.idea|__pycache__|\.aux"`).
- [ ] Remove any remaining superseded artifacts not needed for reproduction.
- [ ] Verify `git ls-files` matches the layout documented in `README.md`.

## 7. Final gate

- [ ] Paper is **accepted** and you are at camera-ready (public release before
      acceptance risks de-anonymization).
- [ ] All BLOCKING items above are checked.
- [ ] Only then: GitHub → repo Settings → Danger Zone → Change visibility →
      Public.

---

*After flipping public, immediately re-verify the anonymization grep on the
public repo — a scrub that was staged locally but not committed is the classic
way a name leaks.*

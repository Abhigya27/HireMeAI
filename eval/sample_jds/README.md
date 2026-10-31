# Sample job descriptions

Put 10-15 real job descriptions here as `.txt`, `.pdf` or `.docx`
(`jd_01.txt` ... `jd_12.txt` to match the skeleton in `../jd_manual_ratings.json`,
or edit the paths there to match your filenames).

- Variance test (`--mode variance`): uses the first 5 files, alphabetically.
- Agreement test (`--mode agreement`): uses the files listed in `../jd_manual_ratings.json`.

Pick a spread: a couple of obvious strong fits, a couple of obvious mismatches,
and several in between. If every JD is the same kind of role the agreement
numbers will look better than they are.

Rate each one in `../jd_manual_ratings.json` **before** running the pipeline on it:
`manual_score` 0-100, and `manual_verdict` one of `strong_fit` (>=75),
`moderate_fit` (50-74), `weak_fit` (<50) -- the same bands the UI uses.

Real JDs are often copyrighted/company-internal; if the repo is public, consider
gitignoring this folder (`eval/sample_jds/*` except this README).

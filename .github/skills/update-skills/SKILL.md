# Skill: Keep Skills Up To Date (HumMobCov)

## Purpose

This skill defines **when and how to update other skill files** in
`.github/skills/`.  Apply it at the end of every prompt session in which
new facts, bugs, fixes, or architecture decisions are discovered.

---

## When to update a skill

Update an existing skill (or create a new one) whenever you discover any of
the following during a prompt session:

| Discovery type | Action |
|---|---|
| A bug is reproduced and fixed | Add to the relevant skill under "Bugs fixed" |
| A bug is identified but **not yet fixed** | Add to "Unresolved issues" in the relevant skill |
| Architecture changes (new method, renamed class, deleted parameter) | Update the relevant structural skill |
| A new execution mode or pipeline path is added | Update `pipeline-store-s3/SKILL.md` |
| A new metric or store kind is added | Update `pipeline-store-s3/SKILL.md` AND `new_project_structure/SKILL.md` |
| A previously "Unresolved" issue is solved | Move it from "Unresolved" to "Bugs fixed" and describe the fix |
| An investigated hypothesis turns out to be false | Record the negative result so it is not re-investigated |
| A naming or path convention is confirmed or changed | Update the relevant skill |
| A library version dependency matters (e.g. polars 1.40.1) | Record it in the relevant skill |

---

## Skill file index

| Skill | File | Covers |
|---|---|---|
| Pipeline, ParquetStore & S3 | `pipeline-store-s3/SKILL.md` | MODE A/B/C execution, store architecture, bugs, S3 config |
| New project structure | `new_project_structure/SKILL.md` | `src/` layout, regions, parameter sets |
| Parallelization | `parallelization/SKILL.md` | numba, polars thread pool, no `n_workers` |
| Paper ↔ pipeline sync | `paper_pipeline_sync/SKILL.md` | figure names, LaTeX ↔ plotter mapping |
| Old pipeline file system | `old_structurefile_system/SKILL.md` | legacy `most_updated_scripts/` structure |
| Old output structure | `old_output_structure/SKILL.md` | legacy `milestones_analysis/` layout |
| Data handling | `data-handling/SKILL.md` | data loading, parquet, geohash, census |
| Keep skills up to date | `update-skills/SKILL.md` | this file |

---

## How to update

1. **Before editing**, read the current skill file with `read_file` to avoid
   duplicate entries and preserve existing structure.
2. Use `replace_string_in_file` for targeted additions (add to an existing
   section).  Use `create_file` only if the skill does not exist yet.
3. Keep entries **concise** — bullet points or short table rows, not prose.
4. For bugs, always record:
   - *Symptom*: what the user observed
   - *Root cause*: the actual code line or logic error
   - *Fix*: what was changed and in which file
5. For unresolved issues, record all investigated-and-ruled-out hypotheses so
   they are not re-investigated.

---

## Session-end checklist

At the end of each prompt session, run through this checklist:

- [ ] Any new bug fixed? → update `pipeline-store-s3/SKILL.md` "Bugs fixed"
- [ ] Any new unresolved error? → update `pipeline-store-s3/SKILL.md` "Unresolved"
- [ ] Any architecture change? → update `new_project_structure/SKILL.md`
- [ ] Any parallelization change? → update `parallelization/SKILL.md`
- [ ] Any figure/paper change? → update `paper_pipeline_sync/SKILL.md`
- [ ] Is a new skill needed for a new domain? → create `new-domain/SKILL.md`
  and add a row to the index table above

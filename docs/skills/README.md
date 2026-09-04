# AXP Skills (schema v1)

AXP loads `*.skill.json` files from `<data-dir>/skills/`. Files are portable configuration and are deliberately
independent of the search database. The JSON `id` is authoritative; filenames only aid humans.

Copy `skill-v1.example.json` into the active directory, rename it, and replace its fictional matching and indexed
path values. Changes are detected before listing or resolving Skills; no restart or watcher is needed.

Validate a directory without starting AXP or an AI model:

```console
python scripts/validate_skills.py --path data/skills
```

Business context and answer recipes guide interpretation and layout only. They are not evidence and cannot relax
AXP's indexed-evidence and citation requirements. Retrieval mode `prefer` searches the configured territory before
global fallback. Mode `strict` never searches globally and fails when none of its configured paths exist in the index.

Schema v1 accepts only `.extension` values, `recent_first` or `all_history` temporal policy, 16–48 maximum documents,
and the `strict` evidence policy. Unsupported schema versions and unknown fields are rejected rather than guessed.

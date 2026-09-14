#!/usr/bin/env python3
"""Teach the OLMo trainer to honour OLMO_ADDITIONAL_CHECKPOINTS_FILE.

WHY THIS EXISTS
---------------
pretrain_experiments/frameworks/olmo/__init__.py::set_additional_checkpoints()
pickles a list of absolute global steps and points the environment variable
OLMO_ADDITIONAL_CHECKPOINTS_FILE at it. Nothing reads that variable.

We verified this against the source rather than inferring it: sbordt/OLMo's
`pretrain-experiments` branch differs from upstream main by exactly two files,
olmo/data/__init__.py and olmo/data/patched_memmap_dataset.py, and the only
OLMO_* variables it reads are OLMO_EXPERIMENT_INSERTIONS_FILE and
OLMO_EXPERIMENT_HDF5_INSERTIONS_FILE. The additional-checkpoints hook lives in
sbordt's private OLMo-Private repo (CLAUDE.md: OLMO_PRIVATE_PATH), which we do
not have.

So `training.additional_checkpoint_steps` is currently a SILENT no-op: the run
trains, exits 0, and writes only the regular interval checkpoint. For the
fine-resolution control rerun that means one checkpoint at step 101000 -- which
is already published as a branch -- for six hours of H100 time.

WHAT IT CHANGES
---------------
Three edits to olmo/train.py, all idempotent:

  1. `import pickle`  (os and logging are already imported; pickle is not)
  2. a module-level loader that reads the pickle into a frozenset
  3. one condition in Trainer.fit()'s unsharded-save block, which becomes
     "every N steps OR in the explicit list"

The regular schedule is untouched, so a run without the env var behaves
exactly as before.

NOTE ON save_interval_unsharded
-------------------------------
The patched condition still requires `save_interval_unsharded is not None`,
because that is the guard OLMo already uses to decide whether unsharded saving
is enabled at all. OLMoFramework.train() always passes it, so this holds for
every run launched through the framework. A bare `scripts/train.py` invocation
that omits it will not get additional checkpoints either.

USAGE
-----
    python internal/asc/patch_olmo_additional_checkpoints.py            # dry run
    python internal/asc/patch_olmo_additional_checkpoints.py --apply

Defaults to $SCRATCH/OLMo/olmo/train.py; pass a path to override. Writes a
.bak beside the file and byte-compiles the result before declaring success.
"""

from __future__ import annotations

import argparse
import difflib
import os
import py_compile
import re
import shutil
import sys
import tempfile

MARKER = "_ADDITIONAL_CHECKPOINT_STEPS"

# --- second, independent fix: scripts/convert_olmo2_to_hf.py ----------------
# That script does
#     from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
# which no longer resolves on the cluster's transformers, so every OLMo ->
# HF conversion dies with ModuleNotFoundError. The name is load-bearing (it is
# instantiated in _write_tokenizer), so it cannot simply be dropped -- but it
# has always been re-exported at the top level, which is the stable path.
#
# This matters beyond the training run: evaluating any control checkpoint goes
# through this converter.
CONVERT_MARKER = "# pretrain-experiments: tolerate both transformers layouts"
CONVERT_OLD = (
    "from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast"
)
CONVERT_NEW = (
    CONVERT_MARKER + "\n"
    "try:\n"
    "    from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast\n"
    "except ModuleNotFoundError:  # newer transformers moved the submodule\n"
    "    from transformers import GPT2TokenizerFast"
)

LOADER = '''
# --- pretrain-experiments: additional checkpoint steps ----------------------
# OLMo can only save every N steps, which cannot express the logarithmic
# schedule the unlearning control needs (1, 2, 3, 5, 8, 13, 21, ...). The
# framework pickles an explicit list of ABSOLUTE global steps and names it in
# OLMO_ADDITIONAL_CHECKPOINTS_FILE; this reads it once at import.
#
# Failure here is reported, never silent: a missing or unreadable file used to
# mean the run produced one checkpoint and looked perfectly healthy.
_ADDITIONAL_CHECKPOINT_STEPS: frozenset = frozenset()
_acs_path = os.environ.get("OLMO_ADDITIONAL_CHECKPOINTS_FILE")
if _acs_path:
    try:
        with open(_acs_path, "rb") as _acs_f:
            _ADDITIONAL_CHECKPOINT_STEPS = frozenset(
                int(_s) for _s in pickle.load(_acs_f)
            )
        if os.environ.get("RANK", "0") == "0":
            print(
                f"[olmo] additional checkpoint steps from {_acs_path}: "
                f"{sorted(_ADDITIONAL_CHECKPOINT_STEPS)}",
                file=sys.stderr,
            )
    except Exception as _acs_exc:  # noqa: BLE001 - must never abort training
        print(
            f"[olmo] WARNING: could not read "
            f"OLMO_ADDITIONAL_CHECKPOINTS_FILE={_acs_path} ({_acs_exc}); "
            f"NO additional checkpoints will be saved",
            file=sys.stderr,
        )
'''

# The line we widen. Anchored on its distinctive text, with the indentation
# captured rather than assumed -- the block sits inside Trainer.fit() and its
# nesting depth is not something to hard-code.
COND_RE = re.compile(
    r"^(?P<indent>[ \t]*)and self\.global_step % self\.cfg\.save_interval_unsharded == 0[ \t]*$",
    re.MULTILINE,
)


def patch(text: str) -> tuple[str, list[str]]:
    """Return (new_text, notes). Raises SystemExit on anything unexpected."""
    notes: list[str] = []

    if MARKER in text:
        return text, ["already patched (marker present); nothing to do"]

    # 1. stdlib imports the loader needs. The fork's train.py imports os and
    #    logging but NOT pickle or sys.
    #
    #    `sys` matters more than it looks. The loader writes to sys.stderr, and a
    #    missing import is a NameError at import time, not a syntax error -- so
    #    py_compile reports success and the failure only appears when
    #    OLMO_ADDITIONAL_CHECKPOINTS_FILE is set, which is precisely the run this
    #    patch exists to enable. The except-branch uses sys.stderr too, so it
    #    cannot even report its own failure. Inserted in reverse order so the
    #    result reads os, pickle, sys.
    for mod in ("sys", "pickle"):
        if re.search(rf"^import {mod}$", text, re.MULTILINE):
            notes.append(f"import {mod}: already present")
            continue
        text, n = re.subn(
            rf"^import os$", f"import os\nimport {mod}", text, count=1, flags=re.MULTILINE
        )
        if n != 1:
            raise SystemExit(
                f"ERROR: could not find a top-level 'import os' to anchor "
                f"'import {mod}' to. The file layout has changed; patch by hand."
            )
        notes.append(f"import {mod}: inserted after 'import os'")

    # 2. the loader, immediately after the module logger.
    anchor = "log = logging.getLogger(__name__)"
    if text.count(anchor) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one '{anchor}', found {text.count(anchor)}. "
            "Patch by hand."
        )
    text = text.replace(anchor, anchor + "\n" + LOADER, 1)
    notes.append("loader: inserted after the module logger")

    # 3. widen the unsharded-save condition.
    matches = list(COND_RE.finditer(text))
    if len(matches) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one unsharded-save condition line, found "
            f"{len(matches)}. Not patching -- inspect Trainer.fit() by hand."
        )
    m = matches[0]
    ind = m.group("indent")
    replacement = (
        f"{ind}and (\n"
        f"{ind}    self.global_step % self.cfg.save_interval_unsharded == 0\n"
        f"{ind}    or self.global_step in {MARKER}\n"
        f"{ind})"
    )
    text = text[: m.start()] + replacement + text[m.end():]
    notes.append("condition: widened to 'every N steps OR in the explicit list'")

    return text, notes


def patch_converter(text: str) -> tuple[str, list[str]]:
    """Make scripts/convert_olmo2_to_hf.py import GPT2TokenizerFast portably.

    Separate from patch() because it touches a different file for a different
    reason: this one is not about checkpoint schedules at all, it is about the
    conversion path that BOTH the training loop and every checkpoint evaluation
    depend on.

    The name is load-bearing -- _write_tokenizer instantiates it -- so the fix
    is a fallback import, not a deletion.
    """
    if CONVERT_MARKER in text:
        return text, ["already patched (marker present); nothing to do"]

    n = text.count(CONVERT_OLD)
    if n != 1:
        raise SystemExit(
            f"ERROR: expected exactly one occurrence of\n    {CONVERT_OLD}\n"
            f"but found {n}. The file has changed; patch by hand."
        )
    text = text.replace(CONVERT_OLD, CONVERT_NEW, 1)
    return text, ["GPT2TokenizerFast: import now falls back to the top level"]


def _process(path: str, patch_fn, apply_it: bool) -> int:
    """Read, patch, diff, syntax-check and optionally write ONE file.

    Each target gets its own backup and its own compile gate, so a failure in
    one cannot leave the other half-written.
    """
    if not os.path.isfile(path):
        print(f"ERROR: no such file: {path}", file=sys.stderr)
        return 2

    with open(path, encoding="utf-8") as f:
        original = f.read()

    patched, notes = patch_fn(original)

    print(f"\ntarget: {path}")
    for n in notes:
        print(f"  - {n}")

    if patched == original:
        return 0

    diff = difflib.unified_diff(
        original.splitlines(keepends=True), patched.splitlines(keepends=True),
        fromfile=path, tofile=path + " (patched)", n=3,
    )
    print("\n" + "".join(diff))

    # Byte-compile the candidate before touching the real file: a patch that
    # produces invalid Python must not reach a six-hour training job.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as tmp:
        tmp.write(patched)
        tmp_path = tmp.name
    try:
        py_compile.compile(tmp_path, doraise=True)
        print("syntax check: OK")
    except py_compile.PyCompileError as exc:
        print(f"syntax check: FAILED\n{exc}", file=sys.stderr)
        return 3
    finally:
        os.unlink(tmp_path)

    if not apply_it:
        print("dry run -- nothing written. Re-run with --apply to write.")
        return 0

    backup = path + ".bak"
    shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as f:
        f.write(patched)
    print(f"written. backup at {backup}")
    return 0


def main() -> int:
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
    default = os.path.join(scratch, "OLMo", "olmo", "train.py")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=default,
                    help=f"olmo/train.py to patch (default: {default})")
    ap.add_argument("--apply", action="store_true",
                    help="write the changes; without it this is a dry run")
    ap.add_argument("--skip-converter", action="store_true",
                    help="patch only olmo/train.py, leaving "
                         "scripts/convert_olmo2_to_hf.py alone")
    args = ap.parse_args()

    # scripts/ sits beside olmo/ in the same checkout, so the converter's path
    # follows from the trainer's rather than needing a second argument.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(args.path)))
    converter = os.path.join(repo, "scripts", "convert_olmo2_to_hf.py")

    targets = [(args.path, patch)]
    if not args.skip_converter:
        if os.path.isfile(converter):
            targets.append((converter, patch_converter))
        else:
            print(f"note: no converter at {converter}; skipping that fix",
                  file=sys.stderr)

    rc = 0
    for target_path, fn in targets:
        rc = _process(target_path, fn, args.apply) or rc

    if args.apply and rc == 0:
        print("\nVerify with the 2-step smoke test: step100001-unsharded and")
        print("step100002-unsharded must appear, and the job log must carry")
        print("'[olmo] additional checkpoint steps from ...'.")
    return rc


if __name__ == "__main__":
    sys.exit(main())

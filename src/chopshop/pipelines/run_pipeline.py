# chopshop/pipeline/run_pipeline.py
# -*- coding: utf-8 -*-
"""
ChopShop Pipeline Runner (free-form, preset-driven)

This runner executes an ordered list of arbitrary steps defined in YAML.
Each step specifies *what* to call and *with which kwargs*. Steps can call
either:
  - ChopShop instance methods via "cs.<subapi>.<method>" (preferred), OR
  - Any importable function via "some.module.path.function"

Design goals:
  * Schema-less pass-through: whatever kwargs are present in YAML are forwarded.
  * Ordered execution: the YAML order *is* the execution order.
  * Chaining: outputs from previous steps can be referenced in later steps.
  * Variables: preset-level vars + CLI overrides usable in any step.
  * Required parameter checks: steps can declare `require: [paramA, ...]`.
  * Friendly templating: {{input}}, {{cwd}}, {{last}}, {{var:x}}, {{env:NAME}},
    and {{pick:artifact.sub.path}} for nested picks from prior results.

Example step:
  - call: cs.text.analyze_with_dictionaries
    with:
      analysis_csv: "{{pick:diar_out.raw_files.csv}}"
      dict_paths:
        - dictionaries/liwc/LIWC-22 Dictionary (2022-01-27).dicx
    require: [analysis_csv, dict_paths]
    save_as: dict_csv

Returns:
  A CSV manifest of inputs, success/failure, and final artifacts (JSON).

Usage:
  python -m chopshop.pipelines.run_pipeline \
    --root_dir /data/study \
    --file_type audio \
    --preset speech_chain \
    --workers 2 \
    --var delimiter="," --vars-file overrides.yaml
"""

from __future__ import annotations

import csv
import json
import os
import inspect
from importlib import import_module, resources
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from ..ChopShop import ChopShop
from chopshop.helpers.find_files import find_files


# ---------------------------------------------------------------------------
# Preset discovery & loading
# ---------------------------------------------------------------------------

def list_presets() -> List[str]:
    """List built-in preset YAMLs shipped inside chopshop.pipeline.presets."""
    try:
        pkg = resources.files("chopshop.pipelines.presets")
    except Exception:
        return []
    out: List[str] = []
    for entry in pkg.iterdir():
        if entry.name.endswith(".yaml"):
            out.append(entry.name[:-5])  # strip .yaml
    out.sort()
    return out


def _read_text_resource(pkg_rel_path: str) -> str:
    """Read a resource file shipped in the wheel into text."""
    pkg = resources.files("chopshop.pipelines.presets")
    with resources.as_file(pkg / pkg_rel_path) as real_path:
        return Path(real_path).read_text(encoding="utf-8")


def load_preset_from_name(name: str) -> Any:
    """Load a built-in preset by name (without .yaml)."""
    try:
        text = _read_text_resource(f"{name}.yaml")
    except FileNotFoundError:
        avail = ", ".join(list_presets())
        raise FileNotFoundError(f"Preset '{name}' not found. Available: {avail}")
    return yaml.safe_load(text) or {}


def load_yaml_file(path: str | Path) -> Any:
    """Load a YAML file from disk into Python structures."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def normalize_preset(obj: Any) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Accept either:
      * a list (treated as steps), OR
      * a mapping with keys: {vars: {...}, steps: [...]}

    Returns:
      (steps_list, vars_dict)
    """
    if isinstance(obj, list):
        return obj, {}
    if isinstance(obj, dict):
        steps = obj.get("steps", [])
        vars_ = obj.get("vars", {}) or {}
        if not isinstance(steps, list):
            raise TypeError("Preset 'steps' must be a list.")
        if not isinstance(vars_, dict):
            raise TypeError("Preset 'vars' must be a mapping.")
        return steps, vars_
    raise TypeError("Preset must be a list of steps or a mapping with keys {vars, steps}.")


# ---------------------------------------------------------------------------
# Variables & placeholder resolution
# ---------------------------------------------------------------------------

def parse_cli_vars(pairs: List[str]) -> Dict[str, Any]:
    """
    Parse KEY=VALUE pairs from CLI. Attempts JSON coercion for lists/bools/numbers.
    Example:
      --var text_cols='["text"]' --var delimiter=","
    """
    out: Dict[str, Any] = {}
    for s in pairs:
        if "=" not in s:
            raise SystemExit(f"--var must be KEY=VALUE, got: {s}")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        # try JSON coercion
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def pick_value(obj: Any, path: str) -> Any:
    """
    Pick nested value by dotted path from dicts or attributes.
    Supports integer segments to index lists/tuples, e.g., "items.0.path".
    """
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            # attribute or index
            if part.isdigit() and hasattr(cur, "__getitem__"):
                cur = cur[int(part)]
            else:
                cur = getattr(cur, part)
    return cur


def resolve_placeholders(x: Any, ctx: Dict[str, Any]) -> Any:
    """
    Recursively resolve placeholders in a value:
      - Strings only are templated; dicts/lists recurse.
      - {{input}}: absolute input path for the current job
      - {{cwd}}: current working directory
      - {{last}}: the most recent saved artifact's value
      - {{var:name}}: lookup in vars bag
      - {{env:NAME}}: environment variable
      - {{pick:artifact.path}}: pick nested value from a saved artifact
    """
    if isinstance(x, str):
        s = x
        # pick first (most specific)
        if s.startswith("{{pick:") and s.endswith("}}"):
            inner = s[len("{{pick:"):-2].strip()  # "artifact.path..."
            art, _, sub = inner.partition(".")
            base = ctx["artifacts"].get(art)
            return pick_value(base, sub) if sub else base

        # variables
        if s.startswith("{{var:") and s.endswith("}}"):
            name = s[len("{{var:"):-2].strip()
            return ctx.get("vars", {}).get(name)

        # environment
        if s.startswith("{{env:") and s.endswith("}}"):
            name = s[len("{{env:"):-2].strip()
            return os.environ.get(name)

        # simple keys (input, cwd, last, or any artifact key)
        if s.startswith("{{") and s.endswith("}}"):
            key = s[2:-2].strip()
            if key == "last":
                return ctx["artifacts"].get("_last")
            # try context globals, then artifacts
            return ctx.get(key) or ctx["artifacts"].get(key)

        return s

    if isinstance(x, dict):
        return {k: resolve_placeholders(v, ctx) for k, v in x.items()}
    if isinstance(x, list):
        return [resolve_placeholders(v, ctx) for v in x]
    return x


# ---------------------------------------------------------------------------
# Callable resolution & kwargs filtering
# ---------------------------------------------------------------------------

def resolve_callable(path: str, cs: ChopShop):
    """
    Resolve a function to call based on a dotted path.
      - If path starts with "cs.", walk attributes on the ChopShop instance.
      - Else, import a module.function and return it.
    """
    if path.startswith("cs."):
        obj: Any = cs
        for part in path[3:].split("."):
            obj = getattr(obj, part)
        if not callable(obj):
            raise TypeError(f"{path} is not callable")
        return obj

    mod, _, attr = path.rpartition(".")
    if not mod:
        raise ValueError(f"Invalid callable path: {path}")
    fn = getattr(import_module(mod), attr)
    if not callable(fn):
        raise TypeError(f"{path} is not callable")
    return fn


def filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    """
    Minimal safety: if the function/method doesn’t accept **kwargs, filter
    to only the parameters it declares. This preserves pass-through feel
    while preventing TypeErrors from YAML typos.
    """
    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs  # accepts **kwargs → pass everything
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

def run_steps_for_input(cs: ChopShop, steps: List[dict], input_path: Path, vars_ctx: Dict[str, Any]) -> dict:
    """
    Execute all steps for a single input file. Returns a manifest row dict:
      {input, ok, error, artifacts}
    """
    # Initialize per-input context
    ctx: Dict[str, Any] = {
        "input": str(input_path),
        "cwd": str(Path.cwd()),
        "vars": dict(vars_ctx),
        "artifacts": {},     # values saved by step name; also mirrored at _last
    }
    row = {"input": str(input_path), "ok": False, "error": "", "artifacts": {}}

    try:
        for idx, step in enumerate(steps, 1):
            call = step.get("call")
            if not call:
                # No-op step; skip quietly so users can comment steps out
                continue

            # Resolve callable (cs.* or module.function)
            fn = resolve_callable(call, cs)

            # Read & resolve kwargs
            raw_kwargs = step.get("with", {}) or {}
            kwargs = resolve_placeholders(raw_kwargs, ctx)

            # Optional `when:` gate; after placeholder resolution
            when = step.get("when", None)
            if when is not None:
                cond = resolve_placeholders(when, ctx)
                if not cond:
                    continue  # skip step if condition is falsy

            # Ensure required params exist and are not empty
            required = step.get("require", []) or []
            missing = [r for r in required if (r not in kwargs) or (kwargs[r] in (None, "", []))]
            if missing:
                raise ValueError(
                    f"Step {call} missing required params: {missing}. "
                    f"Provide them in 'with:' or via variables (e.g., {{var:...}})."
                )

            # Filter kwargs for the function signature (light guardrail)
            kwargs = filter_kwargs_for_callable(fn, kwargs)

            # Execute
            result = fn(**kwargs)

            # If the step requests a nested value with `pick:`
            save_val = result
            if "pick" in step and step["pick"]:
                save_val = pick_value(result, step["pick"])

            # Save artifacts
            save_as = step.get("save_as", "_last")
            ctx["artifacts"][save_as] = save_val
            ctx["artifacts"]["_last"] = save_val

        row["ok"] = True
        row["artifacts"] = ctx["artifacts"]
        return row

    except Exception as e:
        row["ok"] = False
        row["error"] = f"{type(e).__name__}: {e}"
        row["artifacts"] = ctx["artifacts"]
        return row


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="ChopShop pipeline runner (free-form, preset-driven)")
    p.add_argument("--root_dir", required=True, help="Folder to scan for inputs")
    p.add_argument("--file_type", default="audio", choices=["audio", "video", "any"], help="What to discover under --root")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--preset", help="Built-in preset name (found in chopshop.pipelines.presets)")
    g.add_argument("--preset-file", help="Path to a YAML preset on disk")

    # Variables that can be referenced from steps via {{var:name}}
    p.add_argument("--vars-file", action="append", default=[], help="Path to a YAML with variables (can repeat)")
    p.add_argument("--var", action="append", default=[], help='Inline var override KEY=VALUE (can repeat); JSON allowed')

    p.add_argument("--workers", type=int, default=2, help="Parallelism across files")
    p.add_argument("--out-manifest", default="pipeline_manifest.csv", help="CSV manifest path")
    p.add_argument("--list-presets", action="store_true", help="List built-in presets and exit")
    return p


def main():
    args = _build_arg_parser().parse_args()

    # Show built-in presets and exit
    if args.list_presets:
        names = list_presets()
        if not names:
            print("No built-in presets found.")
            return
        print("Built-in presets:")
        for n in names:
            print(" -", n)
        return

    # Load preset definition
    if args.preset:
        preset_obj = load_preset_from_name(args.preset)
    elif args.preset_file:
        preset_obj = load_yaml_file(args.preset_file)
    else:
        avail = ", ".join(list_presets())
        raise SystemExit(f"Provide --preset or --preset-file. Available built-ins: {avail}")

    # Normalize to (steps, vars) and merge CLI vars on top
    steps, vars_from_preset = normalize_preset(preset_obj)

    # Merge variables in precedence order: preset < vars-file(s) < --var
    vars_merged: Dict[str, Any] = dict(vars_from_preset)
    for vf in args.vars_file:
        vars_merged.update(load_yaml_file(vf))
    vars_merged.update(parse_cli_vars(args.var))

    # Discover inputs (absolute paths)
    inputs = find_files(args.root_dir, file_type=args.file_type, recursive=True, absolute=True)
    print(f"[pipeline] Found {len(inputs)} '{args.file_type}' input(s) under {args.root_dir}")

    cs = ChopShop()
    results: List[dict] = []

    # Execute jobs in parallel (per-input)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(run_steps_for_input, cs, steps, p, vars_merged) for p in inputs]
        for fut in as_completed(futs):
            results.append(fut.result())

    # Write manifest CSV with JSON-encoded artifacts
    mf = Path(args.out_manifest)
    mf.parent.mkdir(parents=True, exist_ok=True)
    with mf.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["input", "ok", "error", "artifacts"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "input": r.get("input", ""),
                "ok": r.get("ok", False),
                "error": r.get("error", ""),
                "artifacts": json.dumps(r.get("artifacts", {}), ensure_ascii=False, default=str),
            })

    ok = sum(1 for r in results if r.get("ok"))
    print(f"[pipeline] Done: {ok}/{len(results)} succeeded → {mf}")


if __name__ == "__main__":
    main()

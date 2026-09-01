#!/usr/bin/env python
"""Phase 8 gate: every key in the metrics record is one a source file writes.

``Metrics`` loads ``results/metrics.csv``, adds to it and writes the union back,
so the record is append-only within a directory. A key that is renamed in the
source survives in the file under its old name until the file is deleted, and a
re-run does not remove it. Seven keys were renamed in one change and all seven
survived; they were found by reading the file and removed by hand.

This gate reads the key set of ``results/metrics.csv`` and the key expressions
the sources pass to ``Metrics.set`` and ``Metrics.update``, and requires the two
to agree. It reads the sources as text and imports none of them, so it holds
whatever record is on disk against whatever the pipeline would write next.

Keys are not all literal strings. Stage 5 builds a key from a loop variable,
stage 7 builds one from a format string over a table it has just written, and
``analysis/run_all.py`` builds one from a dictionary it fills in a loop. A scan
for quoted strings alone would report every one of those as unwritten, and a
gate that reports 30 false failures is a gate nobody runs. Each key expression
is therefore resolved symbolically into a sequence of literal segments and
holes, and a hole matches any run of key characters. Where the loop is over a
literal tuple the resolution is exact and the expression yields literal keys;
where it is not, the expression yields a pattern. A pattern is permissive by
construction, so the gate states how many of its keys are matched by pattern and
not by literal, and it refuses to run at all on a pattern that carries no
literal text, since such a pattern would match every key and assert nothing.

The converse direction is checked as well, and it is exact only for literal
keys. A literal key the sources write and the record does not hold means the
stage that writes it did not run. A pattern is satisfied by any one of the keys
it matches, so the loss of one member of a family of eighteen leaves the pattern
matched and this gate silent. Gate 06 covers that case from the other side by
rebuilding the record from an empty directory and comparing the two key sets.
"""

from __future__ import annotations

import ast
import csv
import re
import string
import sys
from pathlib import Path

from gate_lib import ROOT, check, config, finish, gate

gate("gate 08 record")

# The files that may write to the record. Everything that constructs a Metrics
# object lives in one of these three places.
SOURCE_PATTERNS = ("src/*.py", "data/download_data.py", "analysis/*.py")
# The characters recorded keys are drawn from. A hole in a key pattern stands
# for one or more of these.
KEY_CHARACTERS = "[A-Za-z0-9_.]+"
# A pattern must pin at least this many characters of literal text, or it
# asserts too little to be worth matching against.
MINIMUM_LITERAL = 3
# Guard against an expression whose alternatives multiply out without bound.
MAXIMUM_ALTERNATIVES = 512

HOLE = None  # a segment that stands for text the scan cannot resolve


def source_files() -> list[Path]:
    found: list[Path] = []
    for pattern in SOURCE_PATTERNS:
        found += sorted(ROOT.glob(pattern))
    return [path for path in found if path.is_file()]


class Scope:
    """One lexical body: a module, or a function without its nested functions.

    Resolution is scoped so that a variable named ``key`` in one function cannot
    contribute its values to a key expression in another.
    """

    def __init__(self, node) -> None:
        self.node = node
        self.statements: list = []
        self.metrics_names: set[str] = set()
        self.dict_keys: dict[str, list] = {}
        self.variables: dict[str, list] = {}
        self.key_expressions: list = []


def scopes_of(module: ast.Module) -> list[Scope]:
    collected: list[Scope] = []

    def collect(node) -> Scope:
        scope = Scope(node)
        stack = list(ast.iter_child_nodes(node))
        while stack:
            child = stack.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                collected.append(collect(child))
                continue
            scope.statements.append(child)
            stack.extend(ast.iter_child_nodes(child))
        return scope

    collected.append(collect(module))
    return collected


def populate(scope: Scope) -> None:
    """Record what the scope binds and which expressions become record keys."""
    for node in scope.statements:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if (isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == "Metrics"):
                        scope.metrics_names.add(target.id)
                    if isinstance(node.value, ast.Dict):
                        scope.dict_keys.setdefault(target.id, []).extend(
                            k for k in node.value.keys if k is not None)
                    scope.variables.setdefault(target.id, []).append(node.value)
                elif (isinstance(target, ast.Subscript)
                      and isinstance(target.value, ast.Name)):
                    # updates["split_" + part + "_patients"] = ...
                    scope.dict_keys.setdefault(target.value.id, []).append(
                        target.slice)
        elif isinstance(node, ast.AugAssign):
            if (isinstance(node.target, ast.Subscript)
                    and isinstance(node.target.value, ast.Name)):
                scope.dict_keys.setdefault(node.target.value.id, []).append(
                    node.target.slice)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bind_loop(scope, node)

    for node in scope.statements:
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id in scope.metrics_names):
            continue
        if function.attr == "set" and node.args:
            scope.key_expressions.append(node.args[0])
        elif function.attr == "update" and node.args:
            argument = node.args[0]
            if isinstance(argument, ast.Dict):
                scope.key_expressions.extend(
                    k for k in argument.keys if k is not None)
            elif isinstance(argument, ast.Name):
                scope.key_expressions.extend(
                    scope.dict_keys.get(argument.id, []))


def bind_loop(scope: Scope, node) -> None:
    """Bind the loop variable of ``for x in ...`` where the source is knowable.

    Two forms are resolved. A loop over a literal tuple or list of strings binds
    the variable to those strings, which is how stages 5 and 7 walk the three
    parts of the split. A loop over ``d.items()`` binds the first name of the
    target to the key expressions of ``d``, which is how ``analysis/run_all.py``
    writes the per-stage timings.
    """
    if isinstance(node.target, ast.Name):
        if isinstance(node.iter, (ast.Tuple, ast.List)):
            scope.variables.setdefault(node.target.id, []).extend(node.iter.elts)
        else:
            scope.variables.setdefault(node.target.id, []).append(node.iter)
        return
    if not (isinstance(node.target, ast.Tuple) and node.target.elts):
        return
    first = node.target.elts[0]
    if not isinstance(first, ast.Name):
        return
    iterated = node.iter
    if (isinstance(iterated, ast.Call)
            and isinstance(iterated.func, ast.Attribute)
            and iterated.func.attr == "items"
            and isinstance(iterated.func.value, ast.Name)):
        scope.variables.setdefault(first.id, []).extend(
            scope.dict_keys.get(iterated.func.value.id, []))
    else:
        scope.variables.setdefault(first.id, []).append(iterated)


def combine(left: list[list], right: list[list]) -> list[list]:
    joined = [a + b for a in left for b in right]
    return joined[:MAXIMUM_ALTERNATIVES]


def resolve(node, scope: Scope, seen: frozenset) -> list[list]:
    """Every sequence of segments the expression can produce.

    A segment is either a string of literal text or ``HOLE``. The unknown is
    always widened to a hole and never dropped, so the result over-approximates
    what the source writes and the gate cannot fail on a key the source does in
    fact produce.
    """
    if node is None or id(node) in seen:
        return [[HOLE]]
    seen = seen | {id(node)}

    if isinstance(node, ast.Constant):
        return [[node.value]] if isinstance(node.value, str) else [[HOLE]]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return combine(resolve(node.left, scope, seen),
                       resolve(node.right, scope, seen))

    if isinstance(node, ast.JoinedStr):
        parts: list[list] = [[]]
        for piece in node.values:
            parts = combine(parts, resolve(piece, scope, seen))
        return parts

    if isinstance(node, ast.FormattedValue):
        return [[HOLE]]

    if isinstance(node, ast.Name):
        bound = scope.variables.get(node.id)
        if not bound:
            return [[HOLE]]
        alternatives: list[list] = []
        for value in bound:
            alternatives += resolve(value, scope, seen)
        return alternatives[:MAXIMUM_ALTERNATIVES] or [[HOLE]]

    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)):
        segments: list = []
        for text, field, _, _ in string.Formatter().parse(node.func.value.value):
            if text:
                segments.append(text)
            if field is not None:
                segments.append(HOLE)
        return [segments or [HOLE]]

    return [[HOLE]]


def tidy(segments: list) -> list:
    """Merge adjacent literals and collapse runs of holes into one."""
    merged: list = []
    for segment in segments:
        if segment is HOLE:
            if merged and merged[-1] is HOLE:
                continue
            merged.append(HOLE)
        elif merged and merged[-1] is not HOLE:
            merged[-1] = merged[-1] + segment
        else:
            merged.append(segment)
    return merged


def as_pattern(segments: list) -> str:
    return "".join(KEY_CHARACTERS if s is HOLE else re.escape(s)
                   for s in segments)


def as_text(segments: list) -> str:
    return "".join("<?>" if s is HOLE else s for s in segments)


# --- read the record -------------------------------------------------------
check("the metrics record is present", config.METRICS.exists(),
      str(config.METRICS.relative_to(ROOT)))
if not config.METRICS.exists():
    finish()

with open(config.METRICS, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
recorded_order = [row["key"] for row in rows]
recorded = set(recorded_order)
values = {row["key"]: row["value"] for row in rows}
check("the record holds no repeated key",
      len(recorded_order) == len(recorded),
      "{} rows, {} distinct keys".format(len(recorded_order), len(recorded)))

shape = re.compile("^" + KEY_CHARACTERS + "$")
malformed = [key for key in recorded_order if not shape.match(key)]
check("every recorded key is drawn from the character set the patterns assume",
      not malformed, ", ".join(malformed[:5]) if malformed else
      "{} keys".format(len(recorded)))

# --- read what the sources write -------------------------------------------
files = source_files()
check("the source files that write the record were found",
      len(files) >= 10, "{} files".format(len(files)))

literal_keys: dict[str, str] = {}
patterns: list[tuple[str, str, str]] = []   # regex, rendering, origin
unresolved: list[str] = []
parse_failures: list[str] = []

for path in files:
    origin = str(path.relative_to(ROOT)).replace("\\", "/")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as error:
        parse_failures.append("{}: {}".format(origin, error))
        continue
    for scope in scopes_of(tree):
        populate(scope)
        for expression in scope.key_expressions:
            for alternative in resolve(expression, scope, frozenset()):
                segments = tidy(alternative)
                literal = sum(len(s) for s in segments if s is not HOLE)
                if literal < MINIMUM_LITERAL:
                    unresolved.append("{}:{} {}".format(
                        origin, getattr(expression, "lineno", 0),
                        as_text(segments)))
                elif any(s is HOLE for s in segments):
                    patterns.append(
                        ("^" + as_pattern(segments) + "$",
                         as_text(segments), origin))
                else:
                    literal_keys.setdefault(segments[0], origin)

check("every source file parses", not parse_failures,
      "; ".join(parse_failures[:3]) if parse_failures else
      "{} files parsed".format(len(files)))

# A pattern with no literal text would match every key and would let a stale key
# through unnoticed, so the gate treats one as a failure of the scan and not as
# a licence to pass.
check("every key expression resolves to literal text",
      not unresolved, "; ".join(sorted(set(unresolved))[:5]) if unresolved else
      "{} literal keys, {} patterns".format(len(literal_keys), len(patterns)))

compiled = [(re.compile(regex), rendering, origin)
            for regex, rendering, origin in patterns]


def written_by(key: str) -> str | None:
    if key in literal_keys:
        return literal_keys[key]
    for regex, rendering, origin in compiled:
        if regex.match(key):
            return "{} {}".format(origin, rendering)
    return None


# --- the record against the sources ----------------------------------------
orphans = [key for key in recorded_order if written_by(key) is None]
check("every key in the record is written by a source file",
      not orphans,
      "{} of {} keys are written by no source: {}".format(
          len(orphans), len(recorded), ", ".join(orphans[:8]))
      if orphans else
      "{} keys, all written by a source file".format(len(recorded)))

by_pattern = [key for key in recorded_order if key not in literal_keys]
check("most of the record is matched by a literal key and not by a pattern",
      len(by_pattern) * 2 < len(recorded),
      "{} of {} keys matched by pattern".format(len(by_pattern), len(recorded)))

# --- the sources against the record ----------------------------------------
# The converse direction. A literal key a source writes and the record does not
# hold means the stage that writes it did not run, or wrote under a condition
# this record did not meet. Every key in these sources is written on the one
# path through its stage, so the comparison is exact and does not fail on a
# branch that was not taken.
absent = sorted(key for key in literal_keys if key not in recorded)
check("every literal key a source writes is present in the record",
      not absent, "{} absent: {}".format(len(absent), ", ".join(absent[:8]))
      if absent else "{} literal keys, all recorded".format(len(literal_keys)))

idle = [rendering for regex, rendering, _ in compiled
        if not any(regex.match(key) for key in recorded)]
check("every key pattern a source writes matches a recorded key",
      not idle, "; ".join(sorted(set(idle))[:5]) if idle else
      "{} patterns, each matched".format(len(compiled)))

# --- the count the record states about itself ------------------------------
check("the record states its own size correctly",
      lambda: int(float(values["metrics_recorded"])) == len(recorded),
      "metrics_recorded {}, {} keys".format(
          values.get("metrics_recorded", "absent"), len(recorded)))

finish()

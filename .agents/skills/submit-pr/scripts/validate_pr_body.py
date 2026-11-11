#!/usr/bin/env python3
"""Validate a PR body against the repository PR template."""

import argparse
import contextlib
import pathlib
import re
import sys

CHECKBOX_RE = re.compile(r"^\s*[-*+]\s*\[([ xX])\]\s*(.+?)\s*$")
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
PROMPTS = [
    "请提供此Pull Request的预期功能",
    "简要描述测试方案",
    "添加精度和性能验证结果",
    "在这里可以添加任何与这个 Pull Request 相关的其他说明",
]


def read_text(path):
    return pathlib.Path(path).read_text(encoding="utf-8-sig")


def heading_entries(text):
    lines = text.splitlines()
    entries = []
    in_fence = False
    for index, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING_RE.match(line)
        if match:
            entries.append((index, match.group(1), match.group(2)))
    return lines, entries


def extract_headings(text):
    _, entries = heading_entries(text)
    return [(level, name) for _, level, name in entries]


def extract_section(text, heading):
    lines, entries = heading_entries(text)
    for position, (index, _, name) in enumerate(entries):
        if name == heading:
            end = entries[position + 1][0] if position + 1 < len(entries) else len(lines)
            return "\n".join(lines[index + 1 : end])
    return None


def extract_checkboxes(text, heading):
    section = extract_section(text, heading)
    if section is None:
        return None
    items = []
    for line in section.splitlines():
        match = CHECKBOX_RE.match(line)
        if match:
            items.append((match.group(1).lower() == "x", match.group(2).strip()))
    return items


def has_content(text, heading):
    section = extract_section(text, heading)
    if section is None:
        return False
    for line in section.splitlines():
        if line.strip() and not HEADING_RE.match(line) and not CHECKBOX_RE.match(line):
            return True
    return False


def validate(template_path, body_path):
    template = read_text(template_path)
    body = read_text(body_path)
    errors = []
    warnings = []

    template_headings = extract_headings(template)
    body_headings = extract_headings(body)

    position = -1
    for key in template_headings:
        try:
            position = body_headings.index(key, position + 1)
        except ValueError:
            errors.append(f"missing or out-of-order heading: {key[0]} {key[1]}")

    template_h2 = {name for level, name in template_headings if level == "##"}
    for level, name in body_headings:
        if level == "##" and name not in template_h2:
            warnings.append(f"extra H2 heading: {name}")

    for section_name in ("类型", "Checklist:"):
        template_items = extract_checkboxes(template, section_name)
        body_items = extract_checkboxes(body, section_name)
        if template_items is None:
            errors.append(f"template missing section: {section_name}")
            continue
        if body_items is None:
            errors.append(f"body missing section: {section_name}")
            continue
        template_labels = [label for _, label in template_items]
        body_labels = [label for _, label in body_items]
        if body_labels != template_labels:
            errors.append(f"{section_name} items differ from template")
        if section_name == "类型":
            checked = sum(1 for value, _ in body_items if value)
            if checked == 0:
                errors.append("类型 section has no checked item")
            elif checked > 1:
                warnings.append(f"类型 section has {checked} checked items; default is one")

    for section_name in ("描述", "如何测试", "精度和性能验证结果", "其他信息"):
        if not has_content(body, section_name):
            errors.append(f"section has no content: {section_name}")

    for prompt in PROMPTS:
        if prompt in body:
            errors.append(f"template prompt remains: {prompt}")

    return errors, warnings


def main():
    with contextlib.suppress(AttributeError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Validate a PR body against the repository PR template.")
    parser.add_argument(
        "--template",
        default=".gitcode/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md",
        help="Path to the repository PR template.",
    )
    parser.add_argument("--body", required=True, help="Path to the generated PR body.")
    args = parser.parse_args()

    errors, warnings = validate(args.template, args.body)

    print(f"template: {args.template}")
    print(f"body: {args.body}")
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

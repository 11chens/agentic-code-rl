from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
import shutil

from .schemas import TaskSpec, save_task, write_json


@dataclass(frozen=True, slots=True)
class FunctionCase:
    slug: str
    function_name: str
    prompt: str
    buggy_source: str
    fixed_source: str
    public_tests: str
    hidden_tests: str
    tags: tuple[str, ...]


CASE_LIBRARY: list[FunctionCase] = [
    FunctionCase(
        slug="prime_edges",
        function_name="is_prime",
        prompt="Fix prime detection for edge cases.",
        buggy_source="""
def is_prime(n):
    if n == 2:
        return True
    for divisor in range(2, n):
        if n % divisor == 0:
            return False
    return True
""",
        fixed_source="""
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= n:
        if n % divisor == 0:
            return False
        divisor += 2
    return True
""",
        public_tests="""
from buggy_lib import is_prime

def test_common_primes_and_composites():
    assert is_prime(2)
    assert is_prime(3)
    assert not is_prime(4)
""",
        hidden_tests="""
from buggy_lib import is_prime

def test_prime_edges():
    assert not is_prime(1)
    assert not is_prime(0)
    assert not is_prime(-7)
    assert is_prime(97)
""",
        tags=("logic", "edge-case"),
    ),
    FunctionCase(
        slug="factorial_zero",
        function_name="factorial",
        prompt="Fix factorial so it handles zero and rejects negative inputs.",
        buggy_source="""
def factorial(n):
    if n < 0:
        return 0
    result = 1
    for value in range(1, n):
        result *= value
    return result
""",
        fixed_source="""
def factorial(n):
    if n < 0:
        raise ValueError("factorial is undefined for negative values")
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result
""",
        public_tests="""
from buggy_lib import factorial

def test_small_factorials():
    assert factorial(1) == 1
    assert factorial(3) == 6
""",
        hidden_tests="""
import pytest
from buggy_lib import factorial

def test_factorial_zero_and_negative():
    assert factorial(0) == 1
    assert factorial(5) == 120
    with pytest.raises(ValueError):
        factorial(-1)
""",
        tags=("math", "boundary"),
    ),
    FunctionCase(
        slug="median_even",
        function_name="median",
        prompt="Fix median for even-length inputs and empty lists.",
        buggy_source="""
def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid]
""",
        fixed_source="""
def median(values):
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
""",
        public_tests="""
from buggy_lib import median

def test_odd_median():
    assert median([3, 1, 2]) == 2
""",
        hidden_tests="""
import pytest
from buggy_lib import median

def test_even_and_empty_median():
    assert median([10, 2, 4, 8]) == 6
    with pytest.raises(ValueError):
        median([])
""",
        tags=("math", "edge-case"),
    ),
    FunctionCase(
        slug="binary_search_index",
        function_name="binary_search",
        prompt="Fix binary_search so it returns the correct index or -1.",
        buggy_source="""
def binary_search(values, target):
    left = 0
    right = len(values)
    while left < right:
        mid = (left + right) // 2
        if values[mid] == target:
            return mid + 1
        if values[mid] < target:
            left = mid + 1
        else:
            right = mid
    return None
""",
        fixed_source="""
def binary_search(values, target):
    left = 0
    right = len(values) - 1
    while left <= right:
        mid = (left + right) // 2
        if values[mid] == target:
            return mid
        if values[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
        public_tests="""
from buggy_lib import binary_search

def test_binary_search_present():
    assert binary_search([1, 3, 5], 3) == 1
""",
        hidden_tests="""
from buggy_lib import binary_search

def test_binary_search_edges():
    assert binary_search([1, 3, 5], 1) == 0
    assert binary_search([1, 3, 5], 7) == -1
""",
        tags=("algorithm", "indexing"),
    ),
    FunctionCase(
        slug="unique_order",
        function_name="unique_preserve_order",
        prompt="Fix unique_preserve_order so it keeps first-seen order.",
        buggy_source="""
def unique_preserve_order(values):
    return list(set(values))
""",
        fixed_source="""
def unique_preserve_order(values):
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
""",
        public_tests="""
from buggy_lib import unique_preserve_order

def test_unique_no_duplicates():
    assert unique_preserve_order([1, 2, 3]) == [1, 2, 3]
""",
        hidden_tests="""
from buggy_lib import unique_preserve_order

def test_unique_keeps_order():
    assert unique_preserve_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
""",
        tags=("data-structure", "order"),
    ),
    FunctionCase(
        slug="chunk_tail",
        function_name="chunk_list",
        prompt="Fix chunk_list so it keeps the final short chunk and validates size.",
        buggy_source="""
def chunk_list(values, size):
    chunks = []
    for index in range(0, len(values) - size, size):
        chunks.append(values[index:index + size])
    return chunks
""",
        fixed_source="""
def chunk_list(values, size):
    if size <= 0:
        raise ValueError("size must be positive")
    chunks = []
    for index in range(0, len(values), size):
        chunks.append(values[index:index + size])
    return chunks
""",
        public_tests="""
from buggy_lib import chunk_list

def test_even_chunks():
    assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
""",
        hidden_tests="""
import pytest
from buggy_lib import chunk_list

def test_tail_chunk_and_bad_size():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError):
        chunk_list([1], 0)
""",
        tags=("list", "boundary"),
    ),
    FunctionCase(
        slug="safe_divide",
        function_name="safe_divide",
        prompt="Fix safe_divide so division by zero returns the provided default.",
        buggy_source="""
def safe_divide(numerator, denominator, default=None):
    return numerator / denominator
""",
        fixed_source="""
def safe_divide(numerator, denominator, default=None):
    if denominator == 0:
        return default
    return numerator / denominator
""",
        public_tests="""
from buggy_lib import safe_divide

def test_normal_division():
    assert safe_divide(8, 2) == 4
""",
        hidden_tests="""
from buggy_lib import safe_divide

def test_zero_division_default():
    assert safe_divide(8, 0, default="n/a") == "n/a"
""",
        tags=("exception", "boundary"),
    ),
    FunctionCase(
        slug="anagram_normalize",
        function_name="is_anagram",
        prompt="Fix is_anagram so it ignores case and spaces.",
        buggy_source="""
def is_anagram(left, right):
    return sorted(left) == sorted(right)
""",
        fixed_source="""
def is_anagram(left, right):
    def normalize(value):
        return sorted(ch.lower() for ch in value if not ch.isspace())
    return normalize(left) == normalize(right)
""",
        public_tests="""
from buggy_lib import is_anagram

def test_simple_anagram():
    assert is_anagram("listen", "silent")
""",
        hidden_tests="""
from buggy_lib import is_anagram

def test_case_and_space_anagram():
    assert is_anagram("Dormitory", "dirty room")
    assert not is_anagram("abc", "abd")
""",
        tags=("string", "normalization"),
    ),
    FunctionCase(
        slug="parse_int_list",
        function_name="parse_int_list",
        prompt="Fix parse_int_list so it trims whitespace and ignores empty fields.",
        buggy_source="""
def parse_int_list(text):
    return [int(part) for part in text.split(",")]
""",
        fixed_source="""
def parse_int_list(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values
""",
        public_tests="""
from buggy_lib import parse_int_list

def test_plain_csv():
    assert parse_int_list("1,2,3") == [1, 2, 3]
""",
        hidden_tests="""
from buggy_lib import parse_int_list

def test_spaced_and_empty_csv():
    assert parse_int_list("1, 2,, 3,") == [1, 2, 3]
""",
        tags=("string", "parsing"),
    ),
    FunctionCase(
        slug="rotate_modulo",
        function_name="rotate_left",
        prompt="Fix rotate_left so large and empty rotations are handled.",
        buggy_source="""
def rotate_left(values, amount):
    return values[amount:] + values[:amount]
""",
        fixed_source="""
def rotate_left(values, amount):
    if not values:
        return []
    amount = amount % len(values)
    return values[amount:] + values[:amount]
""",
        public_tests="""
from buggy_lib import rotate_left

def test_small_rotation():
    assert rotate_left([1, 2, 3], 1) == [2, 3, 1]
""",
        hidden_tests="""
from buggy_lib import rotate_left

def test_large_and_empty_rotation():
    assert rotate_left([1, 2, 3], 4) == [2, 3, 1]
    assert rotate_left([], 2) == []
""",
        tags=("list", "modulo"),
    ),
]


def create_benchmark(out: Path, repos_out: Path | None = None, count: int = 30, overwrite: bool = False) -> list[Path]:
    out = out.resolve()
    repos_out = (repos_out or out.parent / "repos").resolve()
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; pass overwrite=True to replace generated tasks")
    if repos_out.exists() and overwrite:
        shutil.rmtree(repos_out)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    repos_out.mkdir(parents=True, exist_ok=True)

    task_paths: list[Path] = []
    for index in range(count):
        case = CASE_LIBRARY[index % len(CASE_LIBRARY)]
        task_id = f"task_{index + 1:03d}"
        repo_name = task_id
        repo_dir = repos_out / repo_name
        _write_repo(repo_dir, task_id, case)

        task = TaskSpec(
            id=task_id,
            repo_template=repo_name,
            prompt=case.prompt,
            public_tests=["tests/test_public.py"],
            hidden_tests=["tests/test_hidden.py"],
            max_steps=12,
            tags=list(case.tags),
            metadata={
                "function_name": case.function_name,
                "target_file": "src/buggy_lib.py",
                "expert_patch": {
                    "path": "src/buggy_lib.py",
                    "find": _clean(case.buggy_source),
                    "replace": _clean(case.fixed_source),
                },
                "source_case": case.slug,
            },
        )
        task_path = out / f"{task_id}.json"
        save_task(task_path, task)
        task_paths.append(task_path)

    manifest = {
        "task_count": len(task_paths),
        "tasks_dir": str(out),
        "repos_dir": str(repos_out),
        "default_agent": "scripted",
    }
    write_json(out / "manifest.json", manifest)
    return task_paths


def _write_repo(repo_dir: Path, task_id: str, case: FunctionCase) -> None:
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    (repo_dir / "src").mkdir(parents=True, exist_ok=True)
    (repo_dir / "tests").mkdir(parents=True, exist_ok=True)
    (repo_dir / "src" / "buggy_lib.py").write_text(_clean(case.buggy_source), encoding="utf-8")
    (repo_dir / "tests" / "test_public.py").write_text(_clean(case.public_tests), encoding="utf-8")
    (repo_dir / "tests" / "test_hidden.py").write_text(_clean(case.hidden_tests), encoding="utf-8")
    (repo_dir / "README.md").write_text(
        f"# {task_id}\n\n{case.prompt}\n\nTarget function: `{case.function_name}`.\n",
        encoding="utf-8",
    )


def _clean(text: str) -> str:
    return dedent(text).strip() + "\n"

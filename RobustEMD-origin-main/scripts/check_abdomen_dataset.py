#!/usr/bin/env python3
"""检查腹部数据集预处理文件的一致性。

默认检查 ``data/ABD/ABDOMEN_MR`` 目录下的 ``chaos_MR_T2_normalized`` 与
``supervoxels_5000`` 文件夹，验证每个病例是否同时具备图像、标签和
超体素文件，并报告缺失或多余的条目。

使用示例::

    python scripts/check_abdomen_dataset.py \
        --root /media/wyh/robust/data/ABD/ABDOMEN_CT

可以通过参数覆盖子目录名称或超体素前缀，从而复用于其它数据集。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set


@dataclass
class DatasetEntry:
    """记录单个病例是否具有图像、标签和超体素文件。"""

    has_image: bool = False
    has_label: bool = False
    has_supervoxels: bool = False

    def is_complete(self) -> bool:
        return self.has_image and self.has_label and self.has_supervoxels


SUFFIXES: Sequence[str] = (".nii.gz", ".nii")


def collect_ids(
    files: Iterable[Path],
    prefix: str,
    suffixes: Sequence[str] = SUFFIXES,
) -> Dict[str, Path]:
    """按照前缀/后缀解析文件名，返回 {病例编号: Path} 映射。

    ``suffixes`` 可以同时包含 ``.nii`` 与 ``.nii.gz`` 等扩展名，以
    兼容不同的存储格式。
    """

    result: Dict[str, Path] = {}
    for path in files:
        name = path.name
        if not name.startswith(prefix):
            continue

        for suffix in suffixes:
            if not name.endswith(suffix):
                continue
            case_id = name[len(prefix) : -len(suffix)]
            if case_id:
                canonical = case_id
                if case_id.isdigit():
                    canonical = str(int(case_id))
                result[canonical] = path
            break
    return result


def infer_super_prefix(
    files: Sequence[Path],
    suffixes: Sequence[str] = SUFFIXES,
) -> Optional[str]:
    """尝试从已有文件名推断超体素前缀。"""

    pattern = re.compile(r"^(.*?)(\d+)$")
    for path in files:
        name = path.name
        for suffix in suffixes:
            if not name.endswith(suffix):
                continue
            stem = name[: -len(suffix)]
            match = pattern.match(stem)
            if match:
                return match.group(1)
    return None


def scan_dataset(
    root: Path, normalized_dir: str, super_dir: str, super_prefix: Optional[str]
) -> tuple[Dict[str, DatasetEntry], str]:
    entries: Dict[str, DatasetEntry] = {}

    norm_path = root / normalized_dir
    if not norm_path.exists():
        raise FileNotFoundError(f"未找到 {norm_path}，请确认数据目录是否正确。")

    image_map = collect_ids(norm_path.glob("image_*"), "image_")
    label_map = collect_ids(norm_path.glob("label_*"), "label_")

    super_path = root / super_dir
    if not super_path.exists():
        raise FileNotFoundError(f"未找到 {super_path}，请确认超体素目录是否正确。")

    super_files = sorted(super_path.glob("super*"))
    prefix = super_prefix
    if prefix is None:
        prefix = infer_super_prefix(super_files)
        if prefix is None:
            raise ValueError(
                "未能自动识别超体素文件前缀，请通过 --super-prefix 手动指定。"
            )
    super_map = collect_ids(super_files, prefix)

    all_ids: Set[str] = set(image_map) | set(label_map) | set(super_map)
    for case_id in sorted(all_ids):
        entries[case_id] = DatasetEntry(
            has_image=case_id in image_map,
            has_label=case_id in label_map,
            has_supervoxels=case_id in super_map,
        )

    return entries, prefix


def format_missing(entries: Dict[str, DatasetEntry]) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {"image": [], "label": [], "supervoxels": []}
    for case_id, entry in entries.items():
        if not entry.has_image:
            missing["image"].append(case_id)
        if not entry.has_label:
            missing["label"].append(case_id)
        if not entry.has_supervoxels:
            missing["supervoxels"].append(case_id)
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查腹部数据集预处理文件是否齐全")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/ABD/ABDOMEN_CT"),
        help="数据集根目录，需包含预处理影像与超体素子目录",
    )
    parser.add_argument(
        "--normalized-dir",
        default="sabs_CT_normalized",
        help="存放 image_/label_ 文件的子目录名称（默认 sabs_CT_normalized）",
    )
    parser.add_argument(
        "--supervoxels-dir",
        default="supervoxels_5000",
        help="存放超体素文件的子目录名称",
    )
    parser.add_argument(
        "--super-prefix",
        default=None,
        help="超体素文件名前缀；默认自动推断，也可手动指定",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries, prefix = scan_dataset(
        root=args.root,
        normalized_dir=args.normalized_dir,
        super_dir=args.supervoxels_dir,
        super_prefix=args.super_prefix,
    )

    missing = format_missing(entries)
    total_cases = len(entries)
    complete_cases = sum(entry.is_complete() for entry in entries.values())

    print(f"数据集根目录: {args.root}")
    print(f"检测到病例总数: {total_cases}")
    print(f"使用的超体素前缀: {prefix}")
    print(f"全部文件齐全的病例数: {complete_cases}")

    if complete_cases == total_cases:
        print("所有病例均已同时包含图像、标签和超体素文件。")
        return

    def report(category: str, case_ids: List[str]) -> None:
        if case_ids:
            joined = ", ".join(case_ids)
            print(f"缺少{category}的病例 ({len(case_ids)} 个): {joined}")

    report("图像", missing["image"])
    report("标签", missing["label"])
    report("超体素", missing["supervoxels"])


if __name__ == "__main__":
    main()


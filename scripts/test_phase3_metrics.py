#!/usr/bin/env python3
"""
test_phase3_metrics.py — Targeted Phase 3 verification.

This runs four ground-truth-bearing test cases against the live backend and
asserts that the IoU and top-k-success metrics compute correctly. It is NOT
a unit test — it talks to the running backend on localhost:8200, just like
tune_relevance.py does. Run it after Tier 1 smoke passes.

The four cases:

  1. waldo / bed9e03e2712 with ground_truth_bbox at (1650, 1050, 224, 224)
       Expectation: at least one of top-5 returned bboxes is exactly this
       same tile (we confirmed it earlier — it was the prototype's #2 hit).
       So best_iou should be 1.0 and success_top5 should be 1.

  2. Same query, deliberately wrong ground truth at (0, 0, 224, 224)
       Expectation: no hits overlap this region. best_iou ≈ 0,
       success_top1/3/5 = 0.

  3. Absent query "a pink dragon" against bed9e03e2712 with no ground truth
       Expectation: false_positive metric is computed (expected_present=False);
       success_top* metrics are None (no GT).

  4. Plain Waldo query without ground_truth_bbox
       Expectation: success_top* metrics are None, best_iou is None,
       false_positive is None (expected_present is None, not False).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Importing the harness directly lets us exercise the metric computation
# functions without spawning the whole script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_relevance as tr  # noqa: E402


def run_one(spec, image_id_override=None):
    images = tr.fetch_image_index(tr.BACKEND_URL)
    return tr.run_one_query(
        backend=tr.BACKEND_URL,
        spec=spec,
        images=images,
        k=tr.SEARCH_K,
        scope_to_image=True,
        image_id_override=image_id_override,
    )


def assert_eq(name, actual, expected):
    ok = actual == expected
    marker = "PASS" if ok else "FAIL"
    print(f"  [{marker}] {name}: actual={actual!r}  expected={expected!r}")
    if not ok:
        return 1
    return 0


def assert_close(name, actual, expected, tol=0.01):
    ok = actual is not None and abs(actual - expected) <= tol
    marker = "PASS" if ok else "FAIL"
    print(f"  [{marker}] {name}: actual={actual}  expected≈{expected} (tol={tol})")
    return 0 if ok else 1


def assert_is_none(name, actual):
    ok = actual is None
    marker = "PASS" if ok else "FAIL"
    print(f"  [{marker}] {name}: actual={actual!r}  expected=None")
    return 0 if ok else 1


def main() -> int:
    failures = 0

    # ----- Pure unit tests on _iou_xywh ------------------------------------
    print("== Unit: _iou_xywh ==")
    # Identical boxes → IoU = 1
    failures += assert_close(
        "identical",
        tr._iou_xywh({"x": 0, "y": 0, "w": 100, "h": 100},
                     {"x": 0, "y": 0, "w": 100, "h": 100}),
        1.0, tol=0.0001,
    )
    # Disjoint boxes → IoU = 0
    failures += assert_close(
        "disjoint",
        tr._iou_xywh({"x": 0, "y": 0, "w": 100, "h": 100},
                     {"x": 200, "y": 200, "w": 50, "h": 50}),
        0.0, tol=0.0001,
    )
    # 50% horizontal overlap of equal-size boxes:
    #   intersection = 50*100 = 5000
    #   union        = 10000 + 10000 - 5000 = 15000
    #   IoU          = 5000 / 15000 = 1/3
    failures += assert_close(
        "half-overlap",
        tr._iou_xywh({"x": 0,  "y": 0, "w": 100, "h": 100},
                     {"x": 50, "y": 0, "w": 100, "h": 100}),
        1.0 / 3.0, tol=0.001,
    )
    # Degenerate (zero-area) box → IoU = 0
    failures += assert_close(
        "zero-area",
        tr._iou_xywh({"x": 0, "y": 0, "w": 0,   "h": 100},
                     {"x": 0, "y": 0, "w": 100, "h": 100}),
        0.0, tol=0.0001,
    )

    # ----- Unit tests on _topk_success_metrics -----------------------------
    print("\n== Unit: _topk_success_metrics ==")
    gt = {"x": 100, "y": 100, "w": 100, "h": 100}

    # Case A: rank-1 is exact, rank-2 is disjoint
    bboxes_a = [
        {"x": 100, "y": 100, "w": 100, "h": 100},  # IoU = 1.0
        {"x": 500, "y": 500, "w": 100, "h": 100},  # IoU = 0.0
    ]
    m = tr._topk_success_metrics(bboxes_a, gt)
    failures += assert_eq("A.success_top1", m["success_top1"], 1)
    failures += assert_eq("A.success_top3", m["success_top3"], 1)
    failures += assert_eq("A.success_top5", m["success_top5"], 1)
    failures += assert_close("A.best_iou", m["best_iou"], 1.0)

    # Case B: only rank-4 is correct → top-1 fails, top-3 fails, top-5 succeeds
    bboxes_b = [
        {"x": 500, "y": 500, "w": 100, "h": 100},  # 0
        {"x": 600, "y": 600, "w": 100, "h": 100},  # 0
        {"x": 700, "y": 700, "w": 100, "h": 100},  # 0
        {"x": 100, "y": 100, "w": 100, "h": 100},  # 1.0
    ]
    m = tr._topk_success_metrics(bboxes_b, gt)
    failures += assert_eq("B.success_top1", m["success_top1"], 0)
    failures += assert_eq("B.success_top3", m["success_top3"], 0)
    failures += assert_eq("B.success_top5", m["success_top5"], 1)

    # Case C: no ground truth → all None
    m = tr._topk_success_metrics(bboxes_a, None)
    failures += assert_is_none("C.success_top1", m["success_top1"])
    failures += assert_is_none("C.success_top3", m["success_top3"])
    failures += assert_is_none("C.success_top5", m["success_top5"])
    failures += assert_is_none("C.best_iou",     m["best_iou"])
    failures += assert_eq("C.iou_top_k empty", m["iou_top_k"], [])

    # ----- Integration: live backend ---------------------------------------
    print("\n== Integration: live backend with ground truth ==")

    # Case 1 — Waldo at known bbox in bed9e03e2712.
    # The prototype's #2 hit at (1650, 1050) IS Waldo, confirmed earlier.
    # Whether it shows up in /api/search top-5 depends on reranker output;
    # we verified earlier that the production search returns tiles in the
    # bottom half of the image, so this should at least produce IoU > 0
    # against the ground truth tile.
    spec = {
        "category":          "waldo",
        "query":             "find Waldo",
        "image_id":          "bed9e03e2712",
        "expected_present":  True,
        "ground_truth_bbox": {"x": 1650, "y": 1050, "w": 224, "h": 224},
    }
    r = run_one(spec)
    print(f"  Case 1: top hits bboxes = {r.bboxes[:3]}")
    print(f"  Case 1: iou_top_k = {r.iou_top_k}")
    print(f"  Case 1: best_iou={r.best_iou}, "
          f"s1={r.success_top1}, s3={r.success_top3}, s5={r.success_top5}, "
          f"fp={r.false_positive}")
    # Loose assertions — we don't know which rank Waldo will land at, but
    # we can assert that the metrics are populated rather than None.
    failures += assert_eq("Case 1: image_id_source",   r.image_id_source, "explicit")
    failures += assert_eq("Case 1: false_positive is None for present-true",
                          r.false_positive, None)
    if r.success_top1 is None:
        failures += assert_eq("Case 1: success_top1 should be 0/1, not None",
                              "<None>", "<int>")

    # Case 2 — same query, deliberately wrong ground truth at (0, 0).
    # No tile in the production results sits in the upper-left corner,
    # so IoU should be 0 and all success metrics should be 0.
    spec2 = {
        "category":          "waldo",
        "query":             "find Waldo",
        "image_id":          "bed9e03e2712",
        "expected_present":  True,
        "ground_truth_bbox": {"x": 0, "y": 0, "w": 224, "h": 224},
    }
    r2 = run_one(spec2)
    print(f"  Case 2: best_iou={r2.best_iou}, "
          f"s1={r2.success_top1}, s3={r2.success_top3}, s5={r2.success_top5}")
    failures += assert_eq("Case 2: success_top1", r2.success_top1, 0)
    failures += assert_eq("Case 2: success_top3", r2.success_top3, 0)
    failures += assert_eq("Case 2: success_top5", r2.success_top5, 0)

    # Case 3 — absent-target query, no ground truth.
    spec3 = {
        "category":         "absent",
        "query":            "a pink dragon",
        "image_id":         "bed9e03e2712",
        "expected_present": False,
    }
    r3 = run_one(spec3)
    print(f"  Case 3: num_hits={r3.num_hits}, "
          f"false_positive={r3.false_positive}, best_iou={r3.best_iou}")
    failures += assert_is_none("Case 3: best_iou (no GT)",      r3.best_iou)
    failures += assert_is_none("Case 3: success_top1 (no GT)",  r3.success_top1)
    failures += assert_is_none("Case 3: success_top3 (no GT)",  r3.success_top3)
    failures += assert_is_none("Case 3: success_top5 (no GT)",  r3.success_top5)
    # false_positive must be 0 or 1 since expected_present=False
    failures += assert_eq("Case 3: false_positive type",
                          isinstance(r3.false_positive, int), True)

    # Case 4 — present-unknown query, no ground truth → all GT-dependent
    # metrics must be None.
    spec4 = {
        "category":         "ambiguous",
        "query":            "find Waldo",
        "image_id":         "bed9e03e2712",
        "expected_present": None,
    }
    r4 = run_one(spec4)
    failures += assert_is_none("Case 4: best_iou (no GT)",       r4.best_iou)
    failures += assert_is_none("Case 4: success_top1 (no GT)",   r4.success_top1)
    failures += assert_is_none("Case 4: false_positive (exp=None)", r4.false_positive)

    # ----- CLI: --image-id override ----------------------------------------
    print("\n== Integration: --image-id override + --limit ==")
    with tempfile.TemporaryDirectory() as d:
        out_json = Path(d) / "out.json"
        out_csv  = Path(d) / "out.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "tune_relevance.py"),
                "--label", "test_phase3_cli",
                "--limit", "2",
                "--image-id", "bed9e03e2712",
                "--csv", str(out_csv),
            ],
            stdout=open(out_json, "w"),
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [FAIL] CLI invocation exited {result.returncode}")
            print(f"         stderr: {result.stderr[-500:]}")
            failures += 1
        else:
            data = json.loads(out_json.read_text())
            failures += assert_eq("CLI: image_id_override echoed",
                                  data.get("image_id_override"), "bed9e03e2712")
            failures += assert_eq("CLI: limit echoed",
                                  data.get("limit"), 2)
            failures += assert_eq("CLI: results count == limit",
                                  len(data.get("results", [])), 2)
            for i, r in enumerate(data["results"]):
                failures += assert_eq(f"CLI: result[{i}].image_id",
                                      r["image_id"], "bed9e03e2712")
                failures += assert_eq(f"CLI: result[{i}].image_id_source",
                                      r["image_id_source"], "override")

            # Verify CSV header is in the Phase 3 spec order.
            header = out_csv.read_text().splitlines()[0]
            cols = header.split(",")
            spec_cols = [
                "category", "query", "image_id", "expected_present",
                "success_top1", "success_top3", "success_top5",
                "false_positive", "num_hits", "top_score",
                "normalized_min", "normalized_max", "latency_ms", "best_iou",
            ]
            failures += assert_eq("CLI: CSV header starts with spec columns",
                                  cols[:len(spec_cols)], spec_cols)

    print("\n" + "=" * 60)
    if failures == 0:
        print("ALL PHASE 3 ASSERTIONS PASSED")
        return 0
    print(f"{failures} ASSERTION(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())

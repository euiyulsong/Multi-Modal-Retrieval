import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
# Official Qwen3-VL-Embedding implementation
# ============================================================

DEFAULT_REPO = Path.home() / "Qwen3-VL-Embedding"

if DEFAULT_REPO.exists():
    sys.path.insert(0, str(DEFAULT_REPO))

try:
    from src.models.qwen3_vl_embedding import (
        Qwen3VLEmbedder,
        Qwen3VLForEmbedding,
        Qwen3VLProcessor,
    )
except Exception as e:
    raise RuntimeError(
        "\nOfficial Qwen3-VL-Embedding repo를 import하지 못했습니다.\n\n"
        "실행:\n"
        "git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git "
        "~/Qwen3-VL-Embedding\n"
        "cd ~/Qwen3-VL-Embedding\n"
        "python -m pip install -e .\n\n"
        f"Original error: {e}"
    )


MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

CATEGORIES = [
    "dress",
    "shirt",
    "toptee",
]

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# 모든 query modality에 같은 task instruction 사용.
QUERY_INSTRUCTION = (
    "Retrieve the target fashion product image relevant to the user's input. "
    "The input may contain a reference product image, a description of desired "
    "changes, or both. Use all available information."
)

CACHE_VERSION = "official_qwen3vl_v3"


# ============================================================
# Utils
# ============================================================

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fingerprint(obj):
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha1(raw).hexdigest()[:12]


def safe_name(x):
    return "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in x
    )


# ============================================================
# FashionIQ
# ============================================================

def load_queries(
    root: Path,
    category: str,
    split: str,
):

    path = (
        root
        / "captions"
        / f"cap.{category}.{split}.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    data = read_json(path)

    output = []

    for x in data:

        candidate = x.get("candidate")
        target = x.get("target")

        captions = [
            str(c).strip()
            for c in x.get("captions", [])
            if str(c).strip()
        ]

        if (
            candidate is None
            or target is None
            or not captions
        ):
            continue

        # FashionIQ has two relative captions.
        text = ". ".join(captions)

        output.append(
            {
                "candidate": str(candidate),
                "target": str(target),
                "captions": captions,
                "text": text,
            }
        )

    return output


def load_gallery(
    root: Path,
    category: str,
    split: str,
):

    path = (
        root
        / "image_splits"
        / f"split.{category}.{split}.json"
    )

    data = read_json(path)

    if isinstance(data, dict):

        for key in [
            "images",
            "image_ids",
            "ids",
        ]:
            if key in data:
                data = data[key]
                break

    return list(
        dict.fromkeys(
            str(x)
            for x in data
        )
    )


# ============================================================
# Image index
# ============================================================

def build_image_index(root: Path):

    print(f"Scanning images under {root} ...")

    index = {}

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        if p.suffix.lower() not in IMAGE_EXTS:
            continue

        if p.stem not in index:
            index[p.stem] = str(
                p.resolve()
            )

    print(
        f"Found {len(index):,} images."
    )

    return index


def resolve_paths(
    ids,
    image_index,
    label,
):

    missing = [
        x
        for x in ids
        if x not in image_index
    ]

    if missing:

        print()
        print("=" * 80)
        print(f"MISSING IMAGES: {label}")
        print("=" * 80)

        for x in missing[:30]:
            print(x)

        raise RuntimeError(
            f"{len(missing)} images are missing."
        )

    return [
        image_index[x]
        for x in ids
    ]


# ============================================================
# Checked official model loader
#
# 핵심 수정:
# output_loading_info=True
#
# visual / language weights가 checkpoint에서 빠졌으면
# benchmark 시작 자체를 막는다.
# ============================================================

def load_checked_embedder(
    model_name,
    device,
):

    print()
    print("=" * 80)
    print("LOADING OFFICIAL QWEN3-VL EMBEDDING MODEL")
    print("=" * 80)

    print("model:", model_name)

    model, loading_info = (
        Qwen3VLForEmbedding.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
    )

    missing = loading_info.get(
        "missing_keys",
        [],
    )

    unexpected = loading_info.get(
        "unexpected_keys",
        [],
    )

    mismatched = loading_info.get(
        "mismatched_keys",
        [],
    )

    print()
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))
    print("mismatched keys:", len(mismatched))

    if missing:
        print("\nFirst missing keys:")
        for key in missing[:30]:
            print("  ", key)

    if unexpected:
        print("\nFirst unexpected keys:")
        for key in unexpected[:30]:
            print("  ", key)

    if mismatched:
        print("\nMismatched keys:")
        for x in mismatched[:30]:
            print("  ", x)

    # --------------------------------------------------------
    # Critical weights
    # --------------------------------------------------------

    critical_missing = []

    for key in missing:

        lower = key.lower()

        if (
            "visual" in lower
            or
            "language_model" in lower
            or
            "embed_tokens" in lower
            or
            "patch_embed" in lower
            or
            "pos_embed" in lower
        ):
            critical_missing.append(key)

    if critical_missing:

        print()
        print("CRITICAL MODEL LOADING FAILURE")
        print()

        for key in critical_missing:
            print(key)

        raise RuntimeError(
            "\nCritical Qwen3-VL weights were not loaded.\n"
            "Benchmark aborted to avoid evaluating random weights."
        )

    if mismatched:

        raise RuntimeError(
            "Checkpoint has mismatched parameter shapes. "
            "Benchmark aborted."
        )

    model = model.to(device)
    model.eval()

    processor = Qwen3VLProcessor.from_pretrained(
        model_name,
        padding_side="right",
    )

    # --------------------------------------------------------
    # Create official embedder object without loading model twice
    # --------------------------------------------------------

    embedder = Qwen3VLEmbedder.__new__(
        Qwen3VLEmbedder
    )

    embedder.max_length = 8192

    # official defaults
    embedder.min_pixels = 4096
    embedder.max_pixels = 1843200
    embedder.total_pixels = 7864320
    embedder.fps = 1.0
    embedder.max_frames = 64

    embedder.default_instruction = (
        "Represent the user's input."
    )

    embedder.model = model
    embedder.processor = processor

    # --------------------------------------------------------
    # Parameter sanity
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PARAMETER SANITY")
    print("=" * 80)

    found_visual = False

    for name, param in model.named_parameters():

        if (
            "visual.patch_embed"
            in name
            and "weight" in name
        ):

            found_visual = True

            p = param.detach().float()

            print("parameter:", name)
            print("shape:", tuple(p.shape))
            print("mean:", float(p.mean()))
            print("std:", float(p.std()))
            print("finite:", bool(torch.isfinite(p).all()))

            if not torch.isfinite(p).all():
                raise RuntimeError(
                    "Vision weight contains NaN/Inf."
                )

            if p.std() < 1e-8:
                raise RuntimeError(
                    "Vision patch embedding weight has near-zero variance."
                )

            break

    if not found_visual:
        raise RuntimeError(
            "Could not locate visual.patch_embed weight."
        )

    return embedder


# ============================================================
# Encoding
# ============================================================

@torch.inference_mode()
def encode_batches(
    embedder,
    inputs,
    batch_size,
    desc,
):

    output = []

    for start in tqdm(
        range(
            0,
            len(inputs),
            batch_size,
        ),
        desc=desc,
    ):

        batch = inputs[
            start:start + batch_size
        ]

        emb = embedder.process(
            batch,
            normalize=True,
        )

        emb = (
            emb
            .detach()
            .float()
            .cpu()
        )

        output.append(emb)

    result = torch.cat(
        output,
        dim=0,
    )

    result = F.normalize(
        result,
        dim=-1,
    )

    return result


def load_or_encode(
    embedder,
    inputs,
    cache_path,
    batch_size,
    desc,
    recompute,
):

    if (
        cache_path.exists()
        and not recompute
    ):

        print(
            f"Loading cache: {cache_path}"
        )

        obj = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        return (
            obj["embeddings"]
            .float()
        )

    emb = encode_batches(
        embedder,
        inputs,
        batch_size,
        desc,
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "embeddings":
                emb.to(torch.float16)
        },
        cache_path,
    )

    return emb


# ============================================================
# Model sanity check
# ============================================================

@torch.inference_mode()
def run_sanity_check(
    embedder,
    gallery_paths,
):

    if len(gallery_paths) < 3:
        return

    print()
    print("=" * 80)
    print("EMBEDDING SANITY CHECK")
    print("=" * 80)

    # same image twice + two different images
    inputs = [
        {"image": gallery_paths[0]},
        {"image": gallery_paths[0]},
        {"image": gallery_paths[1]},
        {"image": gallery_paths[2]},
    ]

    emb = encode_batches(
        embedder,
        inputs,
        batch_size=2,
        desc="sanity images",
    )

    sim = emb @ emb.T

    same = float(sim[0, 1])
    diff1 = float(sim[0, 2])
    diff2 = float(sim[0, 3])

    print(f"same-image similarity : {same:.6f}")
    print(f"different image #1   : {diff1:.6f}")
    print(f"different image #2   : {diff2:.6f}")

    if same < 0.99:

        raise RuntimeError(
            "Same image embeddings are unexpectedly inconsistent."
        )

    if same <= max(diff1, diff2):

        raise RuntimeError(
            "Vision embedding sanity check failed."
        )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(ranks):

    ranks = ranks.float()

    return {
        "n":
            len(ranks),

        "R@1":
            float(
                (ranks <= 1)
                .float()
                .mean()
            ),

        "R@5":
            float(
                (ranks <= 5)
                .float()
                .mean()
            ),

        "R@10":
            float(
                (ranks <= 10)
                .float()
                .mean()
            ),

        "R@50":
            float(
                (ranks <= 50)
                .float()
                .mean()
            ),

        "MRR":
            float(
                (1.0 / ranks)
                .mean()
            ),

        "nDCG@10":
            float(
                torch.where(
                    ranks <= 10,
                    1.0
                    / torch.log2(
                        ranks + 1
                    ),
                    torch.zeros_like(
                        ranks
                    ),
                )
                .mean()
            ),
    }


def ranks_from_scores(
    scores,
    candidate_indices,
    target_indices,
):

    scores = scores.clone()

    rows = torch.arange(
        scores.shape[0],
        device=scores.device,
    )

    # FashionIQ reference image 제거
    scores[
        rows,
        candidate_indices,
    ] = -torch.inf

    target_scores = scores[
        rows,
        target_indices,
    ]

    ranks = (
        1
        + (
            scores
            > target_scores[:, None]
        )
        .sum(dim=1)
    )

    return ranks.cpu()


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate(
    queries,
    gallery_ids,
    gallery_emb,
    text_emb,
    image_emb,
    native_emb,
    alphas,
    device,
    batch_size,
):

    gallery_index = {
        x: i
        for i, x
        in enumerate(gallery_ids)
    }

    candidate_indices = torch.tensor(
        [
            gallery_index[
                q["candidate"]
            ]
            for q in queries
        ],
        dtype=torch.long,
    )

    target_indices = torch.tensor(
        [
            gallery_index[
                q["target"]
            ]
            for q in queries
        ],
        dtype=torch.long,
    )

    gallery_gpu = gallery_emb.to(
        device
    )

    outputs = defaultdict(list)

    for start in tqdm(
        range(
            0,
            len(queries),
            batch_size,
        ),
        desc="Ranking",
    ):

        end = min(
            start + batch_size,
            len(queries),
        )

        t = text_emb[
            start:end
        ].to(device)

        i = image_emb[
            start:end
        ].to(device)

        m = native_emb[
            start:end
        ].to(device)

        c = candidate_indices[
            start:end
        ].to(device)

        target = target_indices[
            start:end
        ].to(device)

        # --------------------------------------------
        # text
        # --------------------------------------------

        text_scores = (
            t
            @ gallery_gpu.T
        )

        outputs[
            ("text_only", None)
        ].append(
            ranks_from_scores(
                text_scores,
                c,
                target,
            )
        )

        # --------------------------------------------
        # image
        # --------------------------------------------

        image_scores = (
            i
            @ gallery_gpu.T
        )

        outputs[
            ("image_only", None)
        ].append(
            ranks_from_scores(
                image_scores,
                c,
                target,
            )
        )

        # --------------------------------------------
        # native multimodal
        # --------------------------------------------

        mm_scores = (
            m
            @ gallery_gpu.T
        )

        outputs[
            ("native_multimodal", None)
        ].append(
            ranks_from_scores(
                mm_scores,
                c,
                target,
            )
        )

        # --------------------------------------------
        # weighted score fusion
        # --------------------------------------------

        for alpha in alphas:

            scores = (
                alpha
                * image_scores
                +
                (1.0 - alpha)
                * text_scores
            )

            outputs[
                ("score_fusion", alpha)
            ].append(
                ranks_from_scores(
                    scores,
                    c,
                    target,
                )
            )

    return {
        k: torch.cat(v)
        for k, v in outputs.items()
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES,
    )

    parser.add_argument(
        "--split",
        default="val",
    )

    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--multimodal-batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "./qwen3vl_official_cache"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./qwen3vl_official_results"
        ),
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
    )

    args = parser.parse_args()

    seed_everything()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected."
        )

    device = torch.device("cuda")

    print("=" * 80)
    print("ENVIRONMENT")
    print("=" * 80)

    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    # ========================================================
    # Model
    # ========================================================

    embedder = load_checked_embedder(
        args.model,
        device,
    )

    # ========================================================
    # Data
    # ========================================================

    image_index = build_image_index(
        args.data_root
    )

    args.cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    alphas = np.arange(
        0,
        1 + args.alpha_step / 2,
        args.alpha_step,
    )

    alphas = [
        round(float(x), 4)
        for x in alphas
    ]

    print("alpha:", alphas)

    all_rows = []
    all_ranks = defaultdict(list)

    sanity_done = False

    # ========================================================
    # Category
    # ========================================================

    for category in args.categories:

        print()
        print("#" * 80)
        print(category.upper())
        print("#" * 80)

        queries = load_queries(
            args.data_root,
            category,
            args.split,
        )

        gallery_ids = load_gallery(
            args.data_root,
            category,
            args.split,
        )

        if args.max_queries is not None:
            queries = queries[
                :args.max_queries
            ]

        print(
            "queries:",
            len(queries),
        )

        print(
            "gallery:",
            len(gallery_ids),
        )

        gallery_set = set(
            gallery_ids
        )

        invalid = [
            q
            for q in queries
            if (
                q["candidate"]
                not in gallery_set
                or
                q["target"]
                not in gallery_set
            )
        ]

        if invalid:
            raise RuntimeError(
                f"{len(invalid)} invalid FashionIQ queries."
            )

        # ====================================================
        # Paths
        # ====================================================

        gallery_paths = resolve_paths(
            gallery_ids,
            image_index,
            f"{category}/gallery",
        )

        reference_ids = [
            q["candidate"]
            for q in queries
        ]

        reference_paths = resolve_paths(
            reference_ids,
            image_index,
            f"{category}/reference",
        )

        # sanity once
        if not sanity_done:

            run_sanity_check(
                embedder,
                gallery_paths,
            )

            sanity_done = True

        # ====================================================
        # Cache IDs
        # ====================================================

        gallery_fp = fingerprint(
            gallery_ids
        )

        query_fp = fingerprint(
            [
                (
                    q["candidate"],
                    q["target"],
                    q["text"],
                )
                for q in queries
            ]
        )

        prefix = (
            f"{CACHE_VERSION}_"
            f"{safe_name(args.model)}_"
            f"{category}_"
            f"{args.split}"
        )

        # ====================================================
        # Gallery image embeddings
        #
        # Document에는 instruction 넣지 않음.
        # ====================================================

        gallery_inputs = [
            {
                "image": p,
            }
            for p in gallery_paths
        ]

        gallery_cache = (
            args.cache_dir
            / (
                f"{prefix}_gallery_"
                f"{gallery_fp}.pt"
            )
        )

        gallery_emb = load_or_encode(
            embedder,
            gallery_inputs,
            gallery_cache,
            args.image_batch_size,
            f"{category}: gallery",
            args.recompute,
        )

        # ====================================================
        # Text-only
        # ====================================================

        text_inputs = [
            {
                "text":
                    q["text"],

                "instruction":
                    QUERY_INSTRUCTION,
            }
            for q in queries
        ]

        text_cache = (
            args.cache_dir
            / (
                f"{prefix}_text_"
                f"{query_fp}.pt"
            )
        )

        text_emb = load_or_encode(
            embedder,
            text_inputs,
            text_cache,
            args.text_batch_size,
            f"{category}: text query",
            args.recompute,
        )

        # ====================================================
        # Image-only
        # ====================================================

        image_inputs = [
            {
                "image":
                    path,

                "instruction":
                    QUERY_INSTRUCTION,
            }
            for path in reference_paths
        ]

        image_cache = (
            args.cache_dir
            / (
                f"{prefix}_image_"
                f"{query_fp}.pt"
            )
        )

        image_emb = load_or_encode(
            embedder,
            image_inputs,
            image_cache,
            args.image_batch_size,
            f"{category}: image query",
            args.recompute,
        )

        # ====================================================
        # Native multimodal
        # ====================================================

        native_inputs = [
            {
                "image":
                    image_path,

                "text":
                    q["text"],

                "instruction":
                    QUERY_INSTRUCTION,
            }
            for image_path, q
            in zip(
                reference_paths,
                queries,
            )
        ]

        native_cache = (
            args.cache_dir
            / (
                f"{prefix}_native_"
                f"{query_fp}.pt"
            )
        )

        native_emb = load_or_encode(
            embedder,
            native_inputs,
            native_cache,
            args.multimodal_batch_size,
            f"{category}: native multimodal",
            args.recompute,
        )

        # ====================================================
        # Dimension check
        # ====================================================

        dimensions = {
            gallery_emb.shape[-1],
            text_emb.shape[-1],
            image_emb.shape[-1],
            native_emb.shape[-1],
        }

        print(
            "embedding dims:",
            dimensions,
        )

        if len(dimensions) != 1:

            raise RuntimeError(
                "Embedding dimensions do not match."
            )

        # ====================================================
        # Evaluate
        # ====================================================

        ranks = evaluate(
            queries,
            gallery_ids,
            gallery_emb,
            text_emb,
            image_emb,
            native_emb,
            alphas,
            device,
            args.eval_batch_size,
        )

        for (
            method,
            alpha
        ), r in ranks.items():

            metrics = calculate_metrics(r)

            all_rows.append(
                {
                    "category":
                        category,

                    "method":
                        method,

                    "alpha":
                        np.nan
                        if alpha is None
                        else alpha,

                    **metrics,
                }
            )

            all_ranks[
                (method, alpha)
            ].append(r)

        del (
            gallery_emb,
            text_emb,
            image_emb,
            native_emb,
        )

        torch.cuda.empty_cache()

    # ========================================================
    # ALL categories
    # ========================================================

    for (
        method,
        alpha
    ), values in all_ranks.items():

        r = torch.cat(values)

        all_rows.append(
            {
                "category":
                    "ALL",

                "method":
                    method,

                "alpha":
                    np.nan
                    if alpha is None
                    else alpha,

                **calculate_metrics(r),
            }
        )

    df = pd.DataFrame(
        all_rows
    )

    csv_path = (
        args.output_dir
        / "fashioniq_qwen3vl_official.csv"
    )

    # CSV first.
    # matplotlib 없어도 benchmark 결과는 항상 저장.
    df.to_csv(
        csv_path,
        index=False,
    )

    # ========================================================
    # Print results
    # ========================================================

    result = df[
        df["category"] == "ALL"
    ].sort_values(
        [
            "method",
            "alpha",
        ],
        na_position="first",
    )

    print()
    print("=" * 110)
    print("ALL CATEGORY RESULTS")
    print("=" * 110)

    print(
        result[
            [
                "method",
                "alpha",
                "n",
                "R@1",
                "R@5",
                "R@10",
                "R@50",
                "MRR",
                "nDCG@10",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # Best fusion
    # ========================================================

    fusion = result[
        result["method"]
        == "score_fusion"
    ]

    best = fusion.sort_values(
        [
            "R@10",
            "MRR",
            "R@50",
        ],
        ascending=False,
    ).iloc[0]

    print()
    print("=" * 80)
    print("BEST FUSION")
    print("=" * 80)

    print(
        f"image weight : {best['alpha']:.2f}"
    )

    print(
        f"text weight  : {1.0 - best['alpha']:.2f}"
    )

    print(
        f"R@10         : {best['R@10']:.6f}"
    )

    print(
        f"R@50         : {best['R@50']:.6f}"
    )

    print(
        f"MRR          : {best['MRR']:.6f}"
    )

    print()
    print(
        "Saved:",
        csv_path,
    )


if __name__ == "__main__":
    main()

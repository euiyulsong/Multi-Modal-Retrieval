import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


# ============================================================
# Constants
# ============================================================

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"

CATEGORIES = [
    "dress",
    "shirt",
    "toptee",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# One instruction for ALL query-side methods.
#
# Important:
# Do not give text/image/native different instructions,
# otherwise prompt differences become another experimental variable.
QUERY_PROMPT = (
    "Retrieve the target fashion product image relevant to the user's input."
)


# ============================================================
# General utils
# ============================================================

def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(x: str):
    return "".join(
        c if c.isalnum() or c in "-._" else "_"
        for c in x
    )


def fingerprint(items):
    raw = json.dumps(
        items,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha1(raw).hexdigest()[:12]


# ============================================================
# Optional: download FashionIQ validation images
#
# This solves:
#
# FileNotFoundError:
# Could not find image B009PMCJLW
#
# Dataset:
# royokong/fashioniq_val_imgs
#
# 15,536 val images
# ============================================================

def download_fashioniq_val_images(data_root: Path):

    from datasets import load_dataset

    print(
        "Downloading FashionIQ validation images "
        "from royokong/fashioniq_val_imgs ..."
    )

    ds = load_dataset(
        "royokong/fashioniq_val_imgs",
        split="val",
    )

    print("rows:", len(ds))

    for i, row in enumerate(
        tqdm(ds, desc="Saving FashionIQ images")
    ):

        category = str(row["category"])
        image_id = str(row["id"])
        image = row["img"]

        out_dir = data_root / category
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_path = out_dir / f"{image_id}.jpg"

        if not out_path.exists():
            image.convert("RGB").save(
                out_path,
                quality=95,
            )

    print("FashionIQ validation images ready.")


# ============================================================
# FashionIQ annotation
# ============================================================

def load_queries(
    data_root: Path,
    category: str,
    split: str,
):

    path = (
        data_root
        / "captions"
        / f"cap.{category}.{split}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"FashionIQ annotation not found:\n{path}"
        )

    raw = read_json(path)

    queries = []

    for x in raw:

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
            or len(captions) == 0
        ):
            continue

        # FashionIQ normally provides 2 relative captions.
        #
        # Example:
        # "is sleeveless"
        # "is darker"
        #
        # We join them into one modification description.
        modification = ". ".join(captions)

        queries.append(
            {
                "candidate": str(candidate),
                "target": str(target),
                "captions": captions,
                "text": modification,
            }
        )

    return queries


def load_gallery(
    data_root: Path,
    category: str,
    split: str,
):

    path = (
        data_root
        / "image_splits"
        / f"split.{category}.{split}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"FashionIQ split not found:\n{path}"
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

    # remove duplicate ids while preserving order
    return list(
        dict.fromkeys(
            str(x) for x in data
        )
    )


# ============================================================
# Image indexing
#
# Instead of assuming:
#
# FashionIQ/images/foo.jpg
#
# or
#
# FashionIQ/dress/foo.jpg
#
# scan recursively.
# ============================================================

def build_image_index(data_root: Path):

    print(
        f"Scanning images under {data_root} ..."
    )

    index = {}

    for p in data_root.rglob("*"):

        if not p.is_file():
            continue

        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_id = p.stem

        # first path wins
        if image_id not in index:
            index[image_id] = str(
                p.resolve()
            )

    print(
        f"Found {len(index):,} images."
    )

    return index


def resolve_image_paths(
    image_ids,
    image_index,
):

    missing = [
        x
        for x in image_ids
        if x not in image_index
    ]

    if missing:

        first = "\n".join(
            missing[:20]
        )

        raise FileNotFoundError(
            f"\nMissing {len(missing):,} FashionIQ images.\n"
            f"First missing IDs:\n{first}\n\n"
            "Run:\n"
            "python fashioniq_qwen3vl.py "
            "--data-root ./FashionIQ "
            "--download-val-images\n"
        )

    return [
        image_index[x]
        for x in image_ids
    ]


# ============================================================
# Qwen3-VL embedding
# ============================================================

@torch.inference_mode()
def encode(
    model,
    inputs,
    batch_size,
    prompt=None,
    desc="Encoding",
):

    kwargs = dict(
        sentences=inputs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    if prompt is not None:
        kwargs["prompt"] = prompt

    print(
        f"{desc}: {len(inputs):,}"
    )

    embeddings = model.encode(
        **kwargs
    )

    # Everything from here is evaluated in float32.
    #
    # Keep embeddings on CPU until ranking to reduce VRAM use.
    embeddings = (
        embeddings
        .detach()
        .cpu()
        .float()
    )

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    return embeddings


# ============================================================
# Cache
# ============================================================

def load_or_encode(
    model,
    inputs,
    cache_path,
    batch_size,
    prompt,
    desc,
    recompute=False,
):

    if (
        cache_path.exists()
        and not recompute
    ):

        print(
            f"Loading cache: {cache_path}"
        )

        data = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        return (
            data["embeddings"]
            .float()
        )

    embeddings = encode(
        model=model,
        inputs=inputs,
        batch_size=batch_size,
        prompt=prompt,
        desc=desc,
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # fp16 is enough for cache.
    # We restore float32 before evaluation.
    torch.save(
        {
            "embeddings":
                embeddings.half()
        },
        cache_path,
    )

    return embeddings


# ============================================================
# Ranking metrics
# ============================================================

def calculate_metrics(ranks):

    ranks = ranks.float()

    return {
        "n": int(len(ranks)),

        "R@1":
            (ranks <= 1)
            .float()
            .mean()
            .item(),

        "R@5":
            (ranks <= 5)
            .float()
            .mean()
            .item(),

        "R@10":
            (ranks <= 10)
            .float()
            .mean()
            .item(),

        "R@50":
            (ranks <= 50)
            .float()
            .mean()
            .item(),

        "MRR":
            (1.0 / ranks)
            .mean()
            .item(),

        "nDCG@10":
            torch.where(
                ranks <= 10,
                1.0 / torch.log2(
                    ranks + 1.0
                ),
                torch.zeros_like(ranks),
            )
            .mean()
            .item(),
    }


def ranks_from_scores(
    scores,
    candidate_indices,
    target_indices,
):
    """
    scores:
        [batch, gallery]

    FashionIQ query contains a REFERENCE image.

    The reference image itself exists in the gallery.

    It MUST be masked.

    Otherwise image-only retrieval trivially retrieves
    the query image itself.
    """

    scores = scores.clone()

    rows = torch.arange(
        len(scores),
        device=scores.device,
    )

    # remove reference product from search results
    scores[
        rows,
        candidate_indices,
    ] = -torch.inf

    target_scores = scores[
        rows,
        target_indices,
    ]

    # Rank without full argsort:
    #
    # rank =
    # 1 + number of items whose score > target score

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
#
# Compare:
#
# 1. text_only
# 2. image_only
# 3. native_multimodal
# 4. score_fusion
#
# score_fusion =
#
# alpha * image_score
# +
# (1-alpha) * text_score
#
# alpha = image weight
# ============================================================

@torch.inference_mode()
def evaluate(
    queries,
    gallery_ids,
    gallery_emb,
    text_emb,
    image_emb,
    multimodal_emb,
    alphas,
    device,
    eval_batch_size,
):

    gallery_idx = {
        image_id: i
        for i, image_id
        in enumerate(gallery_ids)
    }

    candidate_indices = torch.tensor(
        [
            gallery_idx[
                q["candidate"]
            ]
            for q in queries
        ],
        dtype=torch.long,
    )

    target_indices = torch.tensor(
        [
            gallery_idx[
                q["target"]
            ]
            for q in queries
        ],
        dtype=torch.long,
    )

    # gallery is only a few thousand per category;
    # keep the entire matrix on 4090.
    gallery_gpu = (
        gallery_emb
        .to(device)
    )

    ranks_by_method = defaultdict(list)

    for start in tqdm(
        range(
            0,
            len(queries),
            eval_batch_size,
        ),
        desc="Ranking",
    ):

        end = min(
            start + eval_batch_size,
            len(queries),
        )

        t = text_emb[
            start:end
        ].to(device)

        i = image_emb[
            start:end
        ].to(device)

        mm = multimodal_emb[
            start:end
        ].to(device)

        candidates = candidate_indices[
            start:end
        ].to(device)

        targets = target_indices[
            start:end
        ].to(device)

        # ----------------------------------------------------
        # TEXT ONLY
        # ----------------------------------------------------

        text_scores = (
            t
            @ gallery_gpu.T
        )

        ranks_by_method[
            ("text_only", None)
        ].append(
            ranks_from_scores(
                text_scores,
                candidates,
                targets,
            )
        )

        # ----------------------------------------------------
        # IMAGE ONLY
        # ----------------------------------------------------

        image_scores = (
            i
            @ gallery_gpu.T
        )

        ranks_by_method[
            ("image_only", None)
        ].append(
            ranks_from_scores(
                image_scores,
                candidates,
                targets,
            )
        )

        # ----------------------------------------------------
        # TRUE / NATIVE MULTIMODAL
        #
        # Qwen3-VL sees:
        #
        # reference image + modification text
        #
        # together.
        # ----------------------------------------------------

        multimodal_scores = (
            mm
            @ gallery_gpu.T
        )

        ranks_by_method[
            ("native_multimodal", None)
        ].append(
            ranks_from_scores(
                multimodal_scores,
                candidates,
                targets,
            )
        )

        # ----------------------------------------------------
        # TEXT + IMAGE SCORE FUSION
        #
        # This directly answers:
        #
        # Is separate retrieval + weighted ranking better
        # than true multimodal encoding?
        #
        # alpha = image weight
        # ----------------------------------------------------

        for alpha in alphas:

            fused_scores = (
                alpha * image_scores
                + (1.0 - alpha)
                * text_scores
            )

            ranks_by_method[
                ("score_fusion", alpha)
            ].append(
                ranks_from_scores(
                    fused_scores,
                    candidates,
                    targets,
                )
            )

    del gallery_gpu

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        k: torch.cat(v)
        for k, v in
        ranks_by_method.items()
    }


# ============================================================
# Plot
# ============================================================

def plot_results(
    df,
    output_path,
):

    import matplotlib.pyplot as plt

    all_df = df[
        df["category"] == "ALL"
    ]

    fusion = (
        all_df[
            all_df["method"]
            == "score_fusion"
        ]
        .sort_values(
            "alpha"
        )
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        fusion["alpha"],
        fusion["R@10"],
        marker="o",
        label="Text/Image score fusion",
    )

    for method in [
        "text_only",
        "image_only",
        "native_multimodal",
    ]:

        row = all_df[
            all_df["method"]
            == method
        ]

        if len(row) == 0:
            continue

        value = float(
            row.iloc[0]["R@10"]
        )

        plt.axhline(
            value,
            linestyle="--",
            label=f"{method}: {value:.3f}",
        )

    plt.xlabel(
        "Image weight α"
    )

    plt.ylabel(
        "Recall@10"
    )

    plt.title(
        "FashionIQ - Qwen3-VL-Embedding-2B"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


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
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--split",
        default="val",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES,
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
        default=4,
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
            "./qwen3vl_cache"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./qwen3vl_results"
        ),
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
    )

    parser.add_argument(
        "--download-val-images",
        action="store_true",
    )

    args = parser.parse_args()

    seed_everything()

    # ========================================================
    # Download FashionIQ images if requested
    # ========================================================

    if args.download_val_images:

        download_fashioniq_val_images(
            args.data_root
        )

    # ========================================================
    # Device
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU not detected. "
            "Qwen3-VL experiment is intended for GPU."
        )

    device = "cuda"

    print()
    print("=" * 80)
    print("ENVIRONMENT")
    print("=" * 80)

    print(
        "torch:",
        torch.__version__,
    )

    print(
        "torch cuda:",
        torch.version.cuda,
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "VRAM GB:",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024 ** 3),
            2,
        ),
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print("=" * 80)
    print("MODEL")
    print("=" * 80)

    print(args.model)

    # Model config itself uses BF16.
    #
    # RTX 4090 has native BF16 support.
    model = SentenceTransformer(
        args.model,
        device=device,
    )

    model.eval()

    print(
        "embedding dimension:",
        model.get_sentence_embedding_dimension(),
    )

    # ========================================================
    # Alpha sweep
    # ========================================================

    alphas = np.arange(
        0.0,
        1.0
        + args.alpha_step / 2,
        args.alpha_step,
    )

    alphas = [
        round(float(x), 4)
        for x in alphas
    ]

    print(
        "alpha:",
        alphas,
    )

    # ========================================================
    # Images
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

    model_name_safe = safe_name(
        args.model
    )

    rows = []

    global_ranks = defaultdict(list)

    # ========================================================
    # Category loop
    # ========================================================

    for category in args.categories:

        print()
        print("#" * 80)
        print(category.upper())
        print("#" * 80)

        queries = load_queries(
            data_root=args.data_root,
            category=category,
            split=args.split,
        )

        gallery_ids = load_gallery(
            data_root=args.data_root,
            category=category,
            split=args.split,
        )

        # optional smoke-test
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

        # Validate annotation consistency
        bad = []

        for q in queries:

            if (
                q["candidate"]
                not in gallery_set
                or
                q["target"]
                not in gallery_set
            ):
                bad.append(q)

        if bad:

            raise RuntimeError(
                f"{len(bad)} query pairs are not "
                "present in gallery split."
            )

        # ====================================================
        # Resolve paths
        # ====================================================

        gallery_paths = resolve_image_paths(
            gallery_ids,
            image_index,
        )

        reference_paths = resolve_image_paths(
            [
                q["candidate"]
                for q in queries
            ],
            image_index,
        )

        texts = [
            q["text"]
            for q in queries
        ]

        # ====================================================
        # Cache fingerprints
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
            f"{model_name_safe}_"
            f"{category}_"
            f"{args.split}"
        )

        # ====================================================
        # 1. Gallery images
        #
        # Documents are IMAGE ONLY.
        # ====================================================

        gallery_inputs = [
            {
                "image": path
            }
            for path in gallery_paths
        ]

        gallery_cache = (
            args.cache_dir
            / (
                f"{prefix}_gallery_"
                f"{gallery_fp}.pt"
            )
        )

        gallery_emb = load_or_encode(
            model=model,
            inputs=gallery_inputs,
            cache_path=gallery_cache,
            batch_size=args.image_batch_size,
            prompt=None,
            desc=f"{category}: gallery images",
            recompute=args.recompute,
        )

        # ====================================================
        # 2. Text-only query
        #
        # Same Qwen3-VL embedding model.
        # Only text branch is given input.
        # ====================================================

        text_inputs = [
            {
                "text": text
            }
            for text in texts
        ]

        text_cache = (
            args.cache_dir
            / (
                f"{prefix}_text_query_"
                f"{query_fp}.pt"
            )
        )

        text_emb = load_or_encode(
            model=model,
            inputs=text_inputs,
            cache_path=text_cache,
            batch_size=args.text_batch_size,
            prompt=QUERY_PROMPT,
            desc=f"{category}: text-only query",
            recompute=args.recompute,
        )

        # ====================================================
        # 3. Image-only query
        #
        # Same model.
        # Reference image only.
        # ====================================================

        image_inputs = [
            {
                "image": path
            }
            for path in reference_paths
        ]

        image_cache = (
            args.cache_dir
            / (
                f"{prefix}_image_query_"
                f"{query_fp}.pt"
            )
        )

        image_emb = load_or_encode(
            model=model,
            inputs=image_inputs,
            cache_path=image_cache,
            batch_size=args.image_batch_size,
            prompt=QUERY_PROMPT,
            desc=f"{category}: image-only query",
            recompute=args.recompute,
        )

        # ====================================================
        # 4. TRUE multimodal query
        #
        # IMPORTANT:
        #
        # Reference image AND modification text are fed through
        # Qwen3-VL together.
        #
        # This is NOT:
        #
        # image_emb + text_emb
        #
        # This is actual multimodal encoder interaction.
        # ====================================================

        multimodal_inputs = [
            {
                "image": image_path,
                "text": text,
            }
            for image_path, text in zip(
                reference_paths,
                texts,
            )
        ]

        multimodal_cache = (
            args.cache_dir
            / (
                f"{prefix}_multimodal_query_"
                f"{query_fp}.pt"
            )
        )

        multimodal_emb = load_or_encode(
            model=model,
            inputs=multimodal_inputs,
            cache_path=multimodal_cache,
            batch_size=args.multimodal_batch_size,
            prompt=QUERY_PROMPT,
            desc=f"{category}: native multimodal query",
            recompute=args.recompute,
        )

        # ====================================================
        # Dimension sanity check
        # ====================================================

        dims = {
            gallery_emb.shape[-1],
            text_emb.shape[-1],
            image_emb.shape[-1],
            multimodal_emb.shape[-1],
        }

        if len(dims) != 1:

            raise RuntimeError(
                f"Embedding dimension mismatch: {dims}"
            )

        print(
            "embedding dim:",
            gallery_emb.shape[-1],
        )

        # ====================================================
        # Evaluate
        # ====================================================

        ranks = evaluate(
            queries=queries,
            gallery_ids=gallery_ids,
            gallery_emb=gallery_emb,
            text_emb=text_emb,
            image_emb=image_emb,
            multimodal_emb=multimodal_emb,
            alphas=alphas,
            device=device,
            eval_batch_size=args.eval_batch_size,
        )

        # ====================================================
        # Metrics
        # ====================================================

        for (
            method,
            alpha
        ), method_ranks in ranks.items():

            metrics = calculate_metrics(
                method_ranks
            )

            rows.append(
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

            global_ranks[
                (method, alpha)
            ].append(
                method_ranks
            )

        # free CPU data before next category
        del (
            gallery_emb,
            text_emb,
            image_emb,
            multimodal_emb,
        )

        torch.cuda.empty_cache()

    # ========================================================
    # Aggregate categories
    # ========================================================

    for (
        method,
        alpha
    ), rank_list in global_ranks.items():

        ranks = torch.cat(
            rank_list
        )

        metrics = calculate_metrics(
            ranks
        )

        rows.append(
            {
                "category": "ALL",

                "method": method,

                "alpha":
                    np.nan
                    if alpha is None
                    else alpha,

                **metrics,
            }
        )

    # ========================================================
    # Results
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    output_csv = (
        args.output_dir
        / "fashioniq_qwen3vl_results.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    aggregate = df[
        df["category"] == "ALL"
    ].copy()

    aggregate = aggregate.sort_values(
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
        aggregate[
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
    # Final comparison
    # ========================================================

    text_result = aggregate[
        aggregate["method"]
        == "text_only"
    ].iloc[0]

    image_result = aggregate[
        aggregate["method"]
        == "image_only"
    ].iloc[0]

    native_result = aggregate[
        aggregate["method"]
        == "native_multimodal"
    ].iloc[0]

    fusion_df = aggregate[
        aggregate["method"]
        == "score_fusion"
    ]

    best_fusion = fusion_df.loc[
        fusion_df["R@10"].idxmax()
    ]

    print()
    print("=" * 80)
    print("BEST COMPARISON")
    print("=" * 80)

    print(
        f"Text only\n"
        f"  R@10 = {text_result['R@10']:.4f}\n"
        f"  MRR  = {text_result['MRR']:.4f}"
    )

    print()

    print(
        f"Image only\n"
        f"  R@10 = {image_result['R@10']:.4f}\n"
        f"  MRR  = {image_result['MRR']:.4f}"
    )

    print()

    print(
        f"Best text/image score fusion\n"
        f"  image alpha = {best_fusion['alpha']:.2f}\n"
        f"  text weight = {1.0 - best_fusion['alpha']:.2f}\n"
        f"  R@10 = {best_fusion['R@10']:.4f}\n"
        f"  MRR  = {best_fusion['MRR']:.4f}"
    )

    print()

    print(
        f"Native multimodal\n"
        f"  R@10 = {native_result['R@10']:.4f}\n"
        f"  MRR  = {native_result['MRR']:.4f}"
    )

    # ========================================================
    # Plot
    # ========================================================

    plot_path = (
        args.output_dir
        / "fashioniq_qwen3vl_r10.png"
    )

    plot_results(
        df,
        plot_path,
    )

    print()
    print(
        "CSV:",
        output_csv,
    )

    print(
        "Plot:",
        plot_path,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


KNOWN_CATEGORIES = {
    "translation",
    "summarization",
    "qa",
    "math_reasoning",
    "rag",
}


def read_dataframe(path: str | Path) -> pd.DataFrame:
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error

    df = pd.DataFrame(records)
    required = {
        "question_id",
        "category",
        "average_accept_length",
        "tokens_per_second",
        "output_tokens",
        "seconds",
        "verify_method"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["tasks"] = df["category"].where(
        df["category"].isin(KNOWN_CATEGORIES), "multi-turn"
    )
    return df


def acceptance_statistics(
    df: pd.DataFrame,
    method:str
) -> pd.DataFrame:

    stats = (
        df.groupby("tasks", sort=True)["average_accept_length"]
        .agg(mean="mean", std="std", samples="count")
        .rename(
            columns={
                "mean": f"{method}_mean",
                "std": f"{method}_std",
                "samples": f"{method}_samples",
            }
        )
    )
    # Append an item-level overall row using all questions.  This follows
    # the same mean and standard-error calculation as each task row.
    values = df["average_accept_length"]
    stats.loc["Avg. Accept.", f"{method}_mean"] = values.mean()
    stats.loc["Avg. Accept.", f"{method}_std"] = values.std(ddof=1)
    stats.loc["Avg. Accept.", f"{method}_samples"] = values.count()
    stats[f"{method}_standard_error"] = (
        stats[f"{method}_std"]
        / stats[f"{method}_samples"].map(math.sqrt)
    )


    result = stats.copy()
    result=result.reset_index()
    # print(result.columns)
    result = pd.concat(
        (
            result[result["tasks"] != "Avg. Accept."],
            result[result["tasks"] == "Avg. Accept."],
        ),
        ignore_index=True,
    )


    result[f"{method}_mean_plus_minus_se"] = result.apply(
        lambda row: (
            f"{row[f'{method}_mean']:.2f}±"
            f"{row[f'{method}_standard_error']:.2f}"
        ),
        axis=1,
    )

    result=result[["tasks",f"{method}_mean_plus_minus_se"]]

    return result.set_index("tasks").T



def acceptance_group_statistics(df: pd.DataFrame) -> pd.DataFrame:

    stats = (
        df.groupby(by=["verify_method","tasks"], sort=True)["average_accept_length"]
        .agg(mean="mean", std="std", samples="count")
        .rename(
            columns={
                "mean": "method_mean",
                "std": "method_std",
                "samples": "method_samples",
            }
        )
    ).reset_index()
    # Append an item-level overall row using all questions.  This follows
    # the same mean and standard-error calculation as each task row.
    # values = df["average_accept_length"]
    # stats.loc["Avg. Accept.", f"{method}_mean"] = values.mean()
    # stats.loc["Avg. Accept.", f"{method}_std"] = values.std(ddof=1)
    # stats.loc["Avg. Accept.", f"{method}_samples"] = values.count()
    stats2 = (
        df.groupby("verify_method", sort=True)["average_accept_length"]
        .agg(mean="mean", std="std", samples="count")
        .rename(
            columns={
                "mean": "method_mean",
                "std": "method_std",
                "samples": "method_samples",
            }
        )
    ).reset_index()


    stats2["tasks"]="Avg. Accept."
    result = pd.concat([stats,stats2],axis=0)

    result[f"method_standard_error"] = (
        result[f"method_std"]
        / result[f"method_samples"].map(math.sqrt)
    )


    result[f"method_mean_plus_minus_se"] = result.apply(
        lambda row: (
            f"{row[f'method_mean']:.2f}±"
            f"{row[f'method_standard_error']:.2f}"
        ),
        axis=1,
    )

    result=result[["verify_method","tasks",f"method_mean_plus_minus_se"]]

    return result.pivot(index='verify_method', columns='tasks', values=f"method_mean_plus_minus_se").reset_index()

def throughput_group_statistics(df):
    stats = (
        df.groupby("verify_method", sort=True)["tokens_per_second"]
        .agg(mean="mean", std="std", samples="count")
        .rename(
            columns={
                "mean": "mean_tokens_per_second",
                "std": "standard_error",
                "samples": "samples",
            }
        )
    ).reset_index()

    stats[f"standard_error_tokens_per_second"] = (
        stats["standard_error"]
        / stats[f"samples"].map(math.sqrt)
    )

    stats["TPS"]=stats.apply(lambda row:f"{row['mean_tokens_per_second']:.2f}±{row['standard_error_tokens_per_second']:.2f}",axis=1)

    return stats[["verify_method","TPS"]]

def throughput_statistics(
    df: pd.DataFrame,
    method: str,
) -> pd.DataFrame:


    mean_speed = df["tokens_per_second"].mean()
    std_speed = df["tokens_per_second"].std(ddof=1)
    samples = len(df)
    row={
            "method": method,
            "mean_tokens_per_second": mean_speed,
            "standard_error_tokens_per_second": std_speed / math.sqrt(samples),
            "aggregate_tokens_per_second": (
                df["output_tokens"].sum() / df["seconds"].sum()
            ),
            "samples": samples,
        }
    return {"TPS":f"{row['mean_tokens_per_second']:.2f}±{row['standard_error_tokens_per_second']:.2f}"}




def bootstrap_aggregate_throughput(
    token_rrsw: pd.DataFrame,
    traversal: pd.DataFrame,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate uncertainty of sum(tokens) / sum(seconds) by paired bootstrap."""
    paired = token_rrsw[["question_id", "output_tokens", "seconds"]].merge(
        traversal[["question_id", "output_tokens", "seconds"]],
        on="question_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_rrsw", "_traversal"),
    )
    if len(paired) != len(token_rrsw) or len(paired) != len(traversal):
        raise ValueError("token_rrsw and traversal must contain the same question_id values")

    rrsw_tokens = paired["output_tokens_rrsw"].to_numpy(dtype=np.float64)
    rrsw_seconds = paired["seconds_rrsw"].to_numpy(dtype=np.float64)
    traversal_tokens = paired["output_tokens_traversal"].to_numpy(dtype=np.float64)
    traversal_seconds = paired["seconds_traversal"].to_numpy(dtype=np.float64)
    sample_count = len(paired)
    rng = np.random.default_rng(seed)
    rrsw_estimates = np.empty(resamples, dtype=np.float64)
    traversal_estimates = np.empty(resamples, dtype=np.float64)

    for index in range(resamples):
        sampled = rng.integers(0, sample_count, size=sample_count)
        rrsw_estimates[index] = (
            rrsw_tokens[sampled].sum() / rrsw_seconds[sampled].sum()
        )
        traversal_estimates[index] = (
            traversal_tokens[sampled].sum()
            / traversal_seconds[sampled].sum()
        )

    relative_improvements = (
        100.0
        * (traversal_estimates - rrsw_estimates)
        / rrsw_estimates
    )
    # The sample SD of bootstrap estimates is the bootstrap standard error of
    # the aggregate estimator.
    return {
        "token_rrsw_se": float(rrsw_estimates.std(ddof=1)),
        "traversal_se": float(traversal_estimates.std(ddof=1)),
        "relative_improvement_se": float(relative_improvements.std(ddof=1)),
    }




import os
def main(path,method) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = read_dataframe(path)
    acceptance_g=acceptance_group_statistics(df)
    throughput_g=throughput_group_statistics(df)
    acceptance_g=pd.merge(acceptance_g,throughput_g,on='verify_method',how='outer')
    acceptance = acceptance_statistics(df,method)
    throughput = throughput_statistics(df,method)
    acceptance["TPS"] = throughput["TPS"]

    return acceptance

def main_g(path) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = read_dataframe(path)
    acceptance_g=acceptance_group_statistics(df)
    throughput_g=throughput_group_statistics(df)
    acceptance_g=pd.merge(acceptance_g,throughput_g,on='verify_method',how='outer')


    return acceptance_g


if __name__ == "__main__":
    path=r'D:\anconda_workspace\ai_infra_vllm_test\uniVer\results'
    result=pd.DataFrame()
    print(os.listdir(path))#'records_vicuna.jsonl',
    # files=[ 'records_rrsw.jsonl', 'records_traveral.jsonl','records_greedy.jsonl', 'records_univer.jsonl',
    #          'llama_rrsw.jsonl', 'llama_traversal.jsonl','llama_greedy.jsonl', 'llama_univer.jsonl']
    # for file in files:
    #     file_path=os.path.join(path,file)
    #     if not os.path.exists(file_path):
    #         continue
    #     method=file.replace(".jsonl",'')
    #     tmp=main(file_path,method)
    #     # print(tmp)
    #     result=pd.concat([result,tmp],axis=0)

    result=main_g(os.path.join(path,"node31.jsonl"))



    result=result[['verify_method','multi-turn','translation','summarization','qa','math_reasoning','rag',
        'Avg. Accept.', 'TPS']]

    print(result.columns)

    print(result)
    #
    result.to_excel("node31.xlsx",index=False)


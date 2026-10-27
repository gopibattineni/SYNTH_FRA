"""SYNTH benchmark — interactive TRTR/TSTR results dashboard."""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from data_loader import (
    ALL_GENERATORS,
    GENERATORS_DIFFUSION,
    GENERATORS_SDV_GAN,
    coverage_frame,
    experiment_status_frame,
    load_all_results,
    metrics_for_task,
    primary_metric,
    summary_long_frame,
    comparisons_long_frame,
)

st.set_page_config(
    page_title="SYNTH Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading Excel results…")
def get_data():
    results = load_all_results()
    summary = summary_long_frame(results)
    comparisons = comparisons_long_frame(results)
    coverage = coverage_frame(results)
    status = experiment_status_frame(results)
    return results, summary, comparisons, coverage, status


def _available_generators(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return []
    found = summary["generator"].dropna().astype(str).unique().tolist()
    return [g for g in ALL_GENERATORS if g in found]


def _apply_filters(summary: pd.DataFrame, metric: str, task_filter: str) -> pd.DataFrame:
    if summary.empty or metric not in summary.columns:
        return pd.DataFrame()

    df = summary.copy()
    if task_filter != "All":
        df = df[df["task_type"] == task_filter.lower()]
    elif metric in metrics_for_task("classification"):
        df = df[df["task_type"] == "classification"]
    elif metric in metrics_for_task("regression"):
        df = df[df["task_type"] == "regression"]

    return df.dropna(subset=[metric])


def overview_heatmap(summary: pd.DataFrame, metric: str, task_filter: str) -> go.Figure:
    df = _apply_filters(summary, metric, task_filter)
    if df.empty:
        return go.Figure()

    pivot = df.pivot_table(
        index="generator",
        columns="dataset",
        values=metric,
        aggfunc="mean",
    )
    order = [g for g in ALL_GENERATORS if g in pivot.index]
    pivot = pivot.reindex(order)

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        labels=dict(color=metric.replace("_", " ")),
        title=f"Utility gap heatmap — {metric.replace('_', ' ')}",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=460)
    return fig


def coverage_heatmap(coverage: pd.DataFrame) -> go.Figure:
    if coverage.empty:
        return go.Figure()

    pivot = coverage.pivot_table(
        index="generator",
        columns="dataset",
        values="available",
        aggfunc="max",
    )
    order = [g for g in ALL_GENERATORS if g in pivot.index]
    pivot = pivot.reindex(order)

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=["#f2f2f2", "#2ecc71"],
        zmin=0,
        zmax=1,
        labels=dict(color="Results available"),
        title="Generator coverage across datasets (8 generators)",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=460)
    return fig


def generator_ranking(summary: pd.DataFrame, metric: str, task_filter: str) -> go.Figure:
    df = _apply_filters(summary, metric, task_filter)
    if df.empty:
        return go.Figure()

    ranked = (
        df.groupby("generator", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=True)
    )
    order = [g for g in ALL_GENERATORS if g in ranked["generator"].tolist()]
    ranked["generator"] = pd.Categorical(ranked["generator"], categories=order, ordered=True)
    ranked = ranked.sort_values("generator")

    fig = px.bar(
        ranked,
        x="generator",
        y=metric,
        color=metric,
        color_continuous_scale="RdYlGn_r",
        title=f"Average {metric.replace('_', ' ')} across datasets (lower is better)",
    )
    fig.update_layout(showlegend=False, height=400, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def trtr_tstr_bars(comparisons: pd.DataFrame, dataset: str, generator: str, metric_base: str) -> go.Figure:
    df = comparisons[
        (comparisons["dataset"] == dataset) & (comparisons["Generator"] == generator)
    ].copy()
    if df.empty:
        return go.Figure()

    trtr_col = f"{metric_base} Mean_TRTR"
    tstr_col = f"{metric_base} Mean_TSTR"
    if trtr_col not in df.columns or tstr_col not in df.columns:
        return go.Figure()

    plot_df = df[["Model", trtr_col, tstr_col]].rename(
        columns={trtr_col: "TRTR", tstr_col: "TSTR"}
    )
    melted = plot_df.melt(id_vars="Model", var_name="Protocol", value_name=metric_base)

    fig = px.bar(
        melted,
        x="Model",
        y=metric_base,
        color="Protocol",
        barmode="group",
        title=f"{dataset} — {generator}: TRTR vs TSTR ({metric_base})",
    )
    fig.update_layout(height=440, xaxis_tickangle=-45, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def main() -> None:
    results, summary, comparisons, coverage, status = get_data()

    st.title("SYNTH Benchmark Dashboard")
    st.caption(
        "TRTR/TSTR utility results from "
        "`Generators/Experiment with utility data leak/diffusion_dataleak` "
        f"— {len(results)} datasets, {len(_available_generators(summary))}/{len(ALL_GENERATORS)} generators loaded."
    )

    loaded = sum(1 for r in results.values() if r.error is None and not r.summary.empty)
    complete = sum(1 for r in results.values() if r.experiment_status == "complete")
    partial = sum(1 for r in results.values() if r.experiment_status == "partial")
    pending = sum(1 for r in results.values() if r.experiment_status == "pending")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Datasets", len(results))
    col2.metric("With results", loaded)
    col3.metric("Complete (8/8)", complete)
    col4.metric("Partial", partial)
    col5.metric("Pending", pending)

    with st.expander("Generator groups"):
        st.markdown(
            f"- **SDV / GAN ({len(GENERATORS_SDV_GAN)}):** {', '.join(GENERATORS_SDV_GAN)}\n"
            f"- **Diffusion ({len(GENERATORS_DIFFUSION)}):** {', '.join(GENERATORS_DIFFUSION)}"
        )

    st.sidebar.header("Filters")
    task_filter = st.sidebar.radio("Task type", ["All", "Classification", "Regression"], index=0)

    if task_filter == "Classification":
        default_metric = "Accuracy_Drop"
        metric_choices = list(metrics_for_task("classification").keys())
    elif task_filter == "Regression":
        default_metric = "R2_Drop"
        metric_choices = list(metrics_for_task("regression").keys())
    else:
        default_metric = "Accuracy_Drop"
        metric_choices = list(metrics_for_task("classification").keys()) + list(
            metrics_for_task("regression").keys()
        )

    metric = st.sidebar.selectbox(
        "Utility metric",
        metric_choices,
        index=metric_choices.index(default_metric) if default_metric in metric_choices else 0,
    )

    tab_status, tab_overview, tab_dataset, tab_ranking, tab_table = st.tabs(
        ["Experiment status", "Overview", "Dataset detail", "Generator ranking", "Data tables"]
    )

    with tab_status:
        st.subheader("Diffusion data-leak experiment progress")
        st.plotly_chart(coverage_heatmap(coverage), use_container_width=True)

        st.markdown("**Per-dataset status**")
        st.dataframe(status, use_container_width=True, hide_index=True)

        pending_rows = status[status["status"].isin(["pending", "partial"])]
        if not pending_rows.empty:
            st.warning(
                "Some notebooks still need to run or are missing generators. "
                "Pending/partial datasets are listed above."
            )

    with tab_overview:
        st.subheader("Cross-dataset utility gap")
        st.plotly_chart(overview_heatmap(summary, metric, task_filter), use_container_width=True)
        st.plotly_chart(generator_ranking(summary, metric, task_filter), use_container_width=True)
        st.markdown(
            "**How to read:** Positive drops mean synthetic training hurt performance vs TRTR. "
            "Lower values indicate generators that preserve more downstream utility."
        )

    with tab_dataset:
        dataset_names = sorted(
            [r.name for r in results.values() if not r.summary.empty],
            key=lambda n: next(r.number for r in results.values() if r.name == n),
        )
        if not dataset_names:
            st.warning("No dataset results loaded yet.")
            return

        selected = st.selectbox("Dataset", dataset_names)
        ds = next(r for r in results.values() if r.name == selected)
        pm = primary_metric(ds.task_type)

        st.caption(
            f"Status: **{ds.experiment_status}** — "
            f"{len(ds.generators)}/{len(ALL_GENERATORS)} generators available"
        )
        if ds.missing_generators:
            st.info(f"Missing: {', '.join(ds.missing_generators)}")

        gens = [g for g in ALL_GENERATORS if g in ds.summary.get("Generator", pd.Series(dtype=str)).astype(str).tolist()]
        if not gens:
            gens = ds.generators
        generator = st.selectbox("Synthetic generator", gens)

        mcols = st.columns(4)
        if not ds.summary.empty and generator in ds.summary["Generator"].astype(str).values:
            row = ds.summary[ds.summary["Generator"].astype(str) == generator].iloc[0]
            task_metrics = metrics_for_task(ds.task_type)
            for i, (mkey, minfo) in enumerate(task_metrics.items()):
                if mkey in row.index:
                    mcols[i % 4].metric(minfo["label"], f"{row[mkey]:.4f}")

        if pm.replace("_Drop", "") in ["Accuracy", "R2"]:
            metric_base = pm.replace("_Drop", "")
        elif ds.task_type == "classification":
            metric_base = "Accuracy"
        else:
            metric_base = "R2"

        st.plotly_chart(
            trtr_tstr_bars(comparisons, selected, generator, metric_base),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Summary (generator-level)**")
            st.dataframe(ds.summary, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**TRTR baseline (real data)**")
            st.dataframe(ds.trtr, use_container_width=True, hide_index=True)
        if not ds.quality.empty:
            st.markdown("**Quality scores**")
            st.dataframe(ds.quality, use_container_width=True, hide_index=True)

    with tab_ranking:
        st.subheader("Best generator per dataset")
        df = _apply_filters(summary, metric, task_filter)
        if df.empty:
            st.info(
                "No data for this metric with the current filters. "
                "Try Classification + Accuracy_Drop or Regression + R2_Drop."
            )
        else:
            idx = df.groupby("dataset", sort=False)[metric].idxmin().dropna()
            best = df.loc[idx][
                ["dataset", "dataset_number", "generator", metric, "task_type"]
            ].sort_values("dataset_number")
            st.dataframe(best, use_container_width=True, hide_index=True)

            win_counts = best["generator"].value_counts().reset_index()
            win_counts.columns = ["generator", "wins"]
            fig = px.bar(
                win_counts,
                x="generator",
                y="wins",
                color="generator",
                title=f"Generator win count (lowest {metric.replace('_', ' ')})",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_table:
        st.subheader("All summary rows")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.download_button(
            "Download summary CSV",
            summary.to_csv(index=False).encode("utf-8"),
            file_name="synth_trtr_tstr_summary.csv",
            mime="text/csv",
        )
        st.subheader("Coverage matrix")
        st.dataframe(coverage, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

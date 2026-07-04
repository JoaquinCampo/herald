from herald.switch_analysis import (
    build_eda_summary,
    cell_mean_r2,
    label_summary,
    pearson_pairwise,
    residualize_by_cells,
)


def test_label_summary_reports_damage_lift_and_zero() -> None:
    rows = [{"dq": 1.0}, {"dq": -1.0}, {"dq": 0.0}, {"dq": 0.5}]
    summary = label_summary(rows)
    assert summary["n"] == 4
    assert summary["frac_damage"] == 0.5
    assert summary["frac_lift"] == 0.25
    assert summary["frac_zero"] == 0.25
    assert summary["frac_major_damage"] == 0.5


def test_cell_mean_r2_detects_cell_explained_variance() -> None:
    rows = [
        {"cell": "a", "dq": 1.0},
        {"cell": "a", "dq": 1.0},
        {"cell": "b", "dq": -1.0},
        {"cell": "b", "dq": -1.0},
    ]
    assert cell_mean_r2(rows, ("cell",)) == 1.0


def test_residualize_by_cells_subtracts_group_mean() -> None:
    rows = [
        {"cell": "a", "dq": 1.0},
        {"cell": "a", "dq": 3.0},
        {"cell": "b", "dq": 10.0},
    ]
    assert residualize_by_cells(rows, ("cell",)) == [-1.0, 1.0, 0.0]


def test_pearson_pairwise_drops_missing_values() -> None:
    corr, n = pearson_pairwise([1.0, 2.0, None, 3.0], [1.0, 2.0, 3.0, 3.0])
    assert n == 3
    assert corr is not None
    assert corr > 0.99


def test_build_eda_summary_ranks_feature_correlations() -> None:
    rows = []
    for i in range(6):
        rows.append(
            {
                "task": "gsm8k",
                "compressor": "snapkv",
                "ratio": 0.5,
                "s": i % 2,
                "dq": float(i),
                "feat__signal": float(i),
                "feat__noise": 1.0,
            }
        )

    summary = build_eda_summary(rows, top_n_features=5)

    assert summary["inventory"]["n_rows"] == 6
    assert summary["inventory"]["n_features"] == 2
    top = summary["feature_vs_damage"]["top_corr_dq"][0]
    assert top["feature"] == "feat__signal"

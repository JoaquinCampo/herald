from herald.switch_visual_report import render_visual_report


def test_render_visual_report_contains_core_sections() -> None:
    summary = {
        "inventory": {
            "n_rows": 10,
            "n_features": 2,
            "by_task": {"gsm8k": 6, "humaneval": 4},
        },
        "labels": {
            "overall": {
                "frac_damage": 0.3,
                "frac_lift": 0.1,
                "frac_zero": 0.6,
                "mean": 0.2,
            },
            "by_task_compressor_ratio": {
                "gsm8k|snapkv|0.25": {"mean": 0.1},
                "gsm8k|snapkv|0.5": {"mean": 0.2},
                "gsm8k|snapkv|0.75": {"mean": 0.3},
                "gsm8k|snapkv|0.875": {"mean": 0.4},
            },
        },
        "cell_structure": {
            "r2_task_compressor_ratio_s": 0.32,
            "r2_compressor_ratio_s": 0.25,
            "r2_task_compressor_ratio": 0.18,
        },
        "feature_vs_damage": {
            "top_corr_dq": [
                {"feature": "feat__entropy", "corr": -0.1, "n": 10}
            ],
            "top_corr_dq_resid_cell": [
                {"feature": "feat__kl_prev", "corr": 0.2, "n": 9}
            ],
        },
    }

    html = render_visual_report(summary)

    assert "Switch Damage Atlas" in html
    assert "Damage Heatmap" in html
    assert "Residual Feature Signal" in html
    assert "10 rows" in html
    assert "kl_prev" in html

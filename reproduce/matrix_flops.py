"""Matrix-product FLOPs of a logged run (2026-10-30). serving_linear_flops_2x counts nn.Linear only, which misses the
direct matmuls of FiBiNet++'s bilinear interaction (V W, then (V W) V^T on W/8 fields of size 8) and of FinalMLP's
multi-head bilinear fusion (x_h^T W_h y_h; the pyramid fusion contracts the wider head first). Totals agree with
torch.utils.flop_counter on the built models (reproduce/profile_flops.py and results/reference/matrix_flops_gpt.csv)."""


def matrix_flops(r):
    f, m = r["serving_linear_flops_2x"], r.get("model", "")
    if m in ("baseline:fibinetpp", "pyramid:p_fibinetpp"):
        F, D = r["input_proj"] // 8, 8
        f += 2 * (F * D * D + F * F * D)
    elif m == "baseline:finalmlp":
        ov = r["overrides"]; H, X, Y = ov["num_heads"], ov["mlp1_hidden_units"][-1], ov["mlp2_hidden_units"][-1]
        f += 2 * (X * Y // H + Y)
    elif m == "pyramid:p_finalmlp":
        pr = r["preset"]; H, X, Y = pr["num_heads"], pr["mlp1_widths"][-1], pr["mlp2_widths"][-1]
        f += 2 * (X * Y // H + (Y if Y // H <= X // H else X))
    return f

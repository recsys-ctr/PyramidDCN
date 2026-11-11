"""Matrix FLOPs of a logged run, as reported in the paper. linear_flops counts the nn.Linear products; the direct
matmuls of FiBiNet++'s bilinear interaction (V W, then (V W) V^T on W/8 groups of size 8) and of FinalMLP's
multi-head bilinear fusion (x_h^T W_h y_h; the pyramidal fusion contracts the wider head first) are added here.
Totals agree with torch.utils.flop_counter on the built models (reproduce/profile_flops.py and
results/reference/matrix_flops.csv)."""


def matrix_flops(r):
    f, m, a = r["linear_flops"], r.get("model", ""), r.get("args", {})
    if m in ("FiBiNetPP", "PyramidFiBiNetPP"):
        F, D = r["input_proj"] // 8, 8
        f += 2 * (F * D * D + F * F * D)
    elif m == "FinalMLP":
        H, X, Y = a["num_heads"], a["mlp1_hidden_units"][-1], a["mlp2_hidden_units"][-1]
        f += 2 * (X * Y // H + Y)
    elif m == "PyramidFinalMLP":
        H, X, Y = a["num_heads"], a["mlp1_widths"][-1], a["mlp2_widths"][-1]
        f += 2 * (X * Y // H + (Y if Y // H <= X // H else X))
    return f

import json

d = json.load(open(r"results\audits\rq2-confounders-20261112.json", encoding="utf-8"))
print("n", d["n_architectures"], "params", d["parameters_range"])
for k in ("rho_target_clean", "rho_target_params", "rho_clean_params"):
    v = d[k]
    print(k, "None" if v is None else round(v, 4))
print()
for m in d["metrics"][:16]:
    print("{:50s} r={:+.3f} c={:+.3f} b={:+.3f} ci=[{:+.3f},{:+.3f}] {}".format(
        m["metric"], m["rho_target"] or 0, m["rho_clean"] or 0, m["rho_clean_params"] or 0,
        m["ci_low"] or 0, m["ci_high"] or 0, m["verdict"]))

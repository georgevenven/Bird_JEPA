def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6   # M-params 
# Faktor = Batches richtig (N // bs, drop_last) / Batches falsch (int(7600 / bs))
N = {"vision": 10000, "language": 9530}
BS = {
    "vision": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "language": [1, 2, 4, 8, 16, 32, 64, 128, 256],
}

for bs in BS["vision"]:
    line = f"{bs}"
    for correct_samples in N:
        if bs in BS[correct_samples]:
            factor = (N[correct_samples] // bs) / int(7600 / bs)
            line += f" {factor:.4f}"
        else:
            line += " -"
    print(line)





# 1 1.3158 1.2539
# 2 1.3158 1.2539
# 4 1.3158 1.2537
# 8 1.3158 1.2537
# 16 1.3158 1.2526
# 32 1.3165 1.2532
# 64 1.3220 1.2542
# 128 1.3220 1.2542
# 256 1.3448 1.2759
# 512 1.3571 -
# 1024 1.2857 -
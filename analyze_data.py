import csv
import math

train_file = r'c:/Users/akash/OneDrive/Desktop/IIT KGP/train_dataset.csv'

rows = []
with open(train_file, mode='r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        rows.append([float(val) for val in r])

extended_header = header[:-1] + ['residence_time', 'temp_diff', 'overall_yield']
extended_rows = []
for r in rows:
    res_time = r[3] / r[0]
    temp_diff = r[4] - r[2]
    extended_rows.append(r[:-1] + [res_time, temp_diff, r[-1]])

print("Analyzing zero yield vs non-zero yield splits:")
for i, col in enumerate(extended_header[:-1]):
    vals = [r[i] for r in extended_rows]
    is_zero = [1 if r[-1] == 0.0 else 0 for r in extended_rows]
    
    sorted_pairs = sorted(zip(vals, is_zero))
    
    best_split = None
    best_impurity = 999.0
    for j in range(1, len(sorted_pairs)):
        if sorted_pairs[j][0] == sorted_pairs[j-1][0]:
            continue
        split_val = (sorted_pairs[j][0] + sorted_pairs[j-1][0]) / 2.0
        
        left_zeros = sum(1 for p in sorted_pairs[:j] if p[1] == 1)
        left_total = j
        right_zeros = sum(1 for p in sorted_pairs[j:] if p[1] == 1)
        right_total = len(sorted_pairs) - j
        
        p_left_zero = left_zeros / left_total
        p_right_zero = right_zeros / right_total
        gini_left = 1.0 - p_left_zero**2 - (1.0 - p_left_zero)**2
        gini_right = 1.0 - p_right_zero**2 - (1.0 - p_right_zero)**2
        weighted_gini = (left_total * gini_left + right_total * gini_right) / len(sorted_pairs)
        
        if weighted_gini < best_impurity:
            best_impurity = weighted_gini
            best_split = (split_val, left_zeros, left_total, right_zeros, right_total)
            
    if best_split:
        val, lz, lt, rz, rt = best_split
        print(f"Feature: {col}")
        print(f"  Best split threshold: {val:.4f}")
        print(f"  Left side: {lz}/{lt} are zero-yield ({(lz/lt)*100:.1f}%)")
        print(f"  Right side: {rz}/{rt} are zero-yield ({(rz/rt)*100:.1f}%)")
        print(f"  Weighted Gini: {best_impurity:.4f}")

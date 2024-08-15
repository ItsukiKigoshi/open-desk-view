import numpy as np

# 元のデータ (x, y)
data = np.array([[853, 201],
                 [940, 430],
                 [288, 445],
                 [319, 627]])

# x の値でソート
sorted_indices = np.argsort(data[:, 0])
sorted_data = data[sorted_indices]

# リストを前後2つに分割
first_half = sorted_data[:2]
second_half = sorted_data[2:]

# y の値でソート
first_half_sorted = first_half[np.argsort(first_half[:, 1])]
second_half_sorted = second_half[np.argsort(second_half[:, 1])]

# 結果の表示
print("First half sorted by y:")
print(first_half_sorted)
print("\nSecond half sorted by y:")
print(second_half_sorted)

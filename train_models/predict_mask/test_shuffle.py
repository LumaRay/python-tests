import numpy as np
from random import shuffle

# x_out = np.array([111111, 222222, 333333, 444444, 555555, 666666, 777777])
x_out = np.zeros((17 * 1024, 1024 ** 2), dtype=np.uint8)
print(x_out)

'''map_idx_orig_to_current = list(range(len(x_out)))
map_idx_orig_to_final = map_idx_orig_to_current.copy()
shuffle(map_idx_orig_to_final)
for orig_idx in range(len(x_out)):
    # sample_idx = permute_positions_list[sample_idx]
    new_sample_pos = map_idx_orig_to_current[map_idx_orig_to_final[orig_idx]]
    x_out[new_sample_pos], x_out[orig_idx] = x_out[orig_idx], x_out[new_sample_pos]
    map_idx_orig_to_current[orig_idx], map_idx_orig_to_current[new_sample_pos] = new_sample_pos, orig_idx

print(map_idx_orig_to_final)
print(map_idx_orig_to_current)
print(x_out)'''

np.random.shuffle(x_out)
print(x_out)

'''
    permute_positions_list = list(range(len(x_out)))
    permute_shuffled_list = permute_positions_list.copy()
    shuffle(permute_shuffled_list)
    for sample_idx in range(len(x_out)):
        new_sample_pos = permute_shuffled_list[sample_idx]
        new_sample_pos = permute_positions_list[new_sample_pos]
        x_out[new_sample_pos], x_out[sample_idx] = x_out[sample_idx], x_out[new_sample_pos]
        y_out[new_sample_pos], y_out[sample_idx] = y_out[sample_idx], y_out[new_sample_pos]
        permute_positions_list[sample_idx], permute_positions_list[new_sample_pos] = new_sample_pos, sample_idx'''
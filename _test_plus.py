import numpy as np
import itertools

def get_permutations_with_steps(steps, reversed=False):
    if steps == 0:
        return combinations_list
    valid_cobminations = []
    for combination in combinations_list:
        # combination = list(combination)
        max_lvl = 0
        lvl_steps = 0
        for idx in range(6):
            if combination[idx] > max_lvl:
                max_lvl = combination[idx]
                lvl_steps += 1
        if steps == lvl_steps:
            if reversed:
                valid_cobminations += [tuple(list(combination)[::-1])]
            else:
                valid_cobminations += [combination]
    return valid_cobminations

def get_row_permutations(clues, row):
    row_permutations1 = get_permutations_with_steps(clues[row + 6], True)
    row_permutations2 = get_permutations_with_steps(clues[23 - row])
    row_permutations = set(row_permutations1) & set(row_permutations2)
    row_permutations = tuple(row_permutations)
    row_permutations = np.array([np.array(p, dtype=np.int8) for p in row_permutations])
    return row_permutations

def get_col_permutations(clues, col):
    col_permutations1 = get_permutations_with_steps(clues[col])
    col_permutations2 = get_permutations_with_steps(clues[17 - col], True)
    col_permutations = set(col_permutations1) & set(col_permutations2)
    col_permutations = tuple(col_permutations)
    col_permutations = np.array([np.array(p, dtype=np.int8).T for p in col_permutations])
    return col_permutations

def get_permutations(clues, clue_idx):
    if clue_idx < 6:
        return get_col_permutations(clues, clue_idx)
    return get_row_permutations(clues, clue_idx - 6)

def walk_through_permutations_for_clue(district, permutations_clues_matrix, clue_idx):
    permutations = permutations_clues_matrix[clue_idx]
    for permutation_idx, permutation in enumerate(permutations):
        if clue_idx < 1:
            print("Trying level", clue_idx, "permutation", permutation_idx, "/", len(permutations))
        district_copy = district  # .copy()
        no_collisions = True
        # if clue_idx < 6:
        #     test = np.subtract(district_copy, np.expand_dims(permutation, -1))
        #     # test = np.subtract(district_copy[:,:clue_idx + 1], np.expand_dims(permutation, -1))
        #     test_count = np.count_nonzero(test == 0)
        #     if test_count > 0:
        #         no_collisions = False
        # else:
        #     row = clue_idx - 6
        #     if not (district_copy[row] == permutation).all():
        #         no_collisions = False
        for perm_element_idx, perm_element in enumerate(permutation):
            if clue_idx < 6:
                row, col = perm_element_idx, clue_idx
                for idx in range(col):
                    if district_copy[row,idx] == perm_element:
                        no_collisions = False
                        break
            else:
                row, col = clue_idx - 6, perm_element_idx
                if district_copy[row,col] != perm_element:
                    no_collisions = False
                    break
        if not no_collisions:
            continue
        if clue_idx < 6:
            col = clue_idx
            district_copy[:,col] = permutation  # .T
        if clue_idx == 11:
            return district_copy
        district_copy = walk_through_permutations_for_clue(district_copy, permutations_clues_matrix, clue_idx + 1)
        if district_copy is not None:
            return district_copy
    return None

combinations_list = list(itertools.permutations(range(1, 7)))

def solve_puzzle(clues):
    district = np.array([[-1] * 6 for _ in range(6)], dtype=np.int8)

    permutations_clues_matrix = []
    for clue_idx in range(12):
        permutations_clues_matrix += [get_permutations(clues, clue_idx)]


    # # test
    # permutations_0 = get_permutations(clues, 0)
    # permutations_1 = get_permutations(clues, 1)
    # # tests_cleared = np.zeros((0, 6), dtype=np.int8)
    # tests_cleared = np.zeros((0, 2, 6), dtype=np.int8)
    # for perm in permutations_1:
    #     test = np.subtract(permutations_0, perm)
    #     test_cleared = permutations_0[np.all(test != 0, axis=1)]
    #     # tests_cleared = np.vstack((tests_cleared, test_cleared))
    #     # test_cleared = np.stack((test_cleared, perm), axis=1)
    #     # test_cleared = np.stack((test_cleared,  np.expand_dims(perm, -1)), axis=1)
    #     # test_cleared = np.dstack((np.expand_dims(test_cleared, 1),  np.repeat(np.expand_dims(np.expand_dims(perm, 0), 0), test_cleared.shape[0], axis=0)))
    #     test_cleared = np.hstack((np.expand_dims(test_cleared, 1),  np.repeat(np.expand_dims(np.expand_dims(perm, 0), 0), test_cleared.shape[0], axis=0)))
    #     tests_cleared = np.vstack((tests_cleared, test_cleared))
    #     # tests_cleared = np.stack((tests_cleared, test_cleared), axis=0)
    #     # test_cleared = np.hstack((test_cleared, np.expand_dims(perm, -1)))
    #     pass

    np_source = np.expand_dims(np.expand_dims(get_permutations(clues, 0), 0), -2)
    for col in range(1, 6):
        permutations = get_permutations(clues, col)
        tests_cleared = np.zeros((0, col + 1, 6), dtype=np.int8)
        for perm in permutations:
            test = np.subtract(np_source, perm)
            test_idxs = np.all(test != 0, axis=-1)  # , keepdims=True)
            test_idxs = np.all(test_idxs != False, axis=-1)  # , keepdims=True)
            test_cleared = np_source[test_idxs]  # .reshape((len() // np_source[1] // np_source[2], np_source[1], np_source[2]))
            test_cleared_stacked = np.hstack((test_cleared,  np.repeat(np.expand_dims(np.expand_dims(perm, 0), 0), test_cleared.shape[0], axis=0)))
            tests_cleared = np.vstack((tests_cleared, test_cleared_stacked))
            pass
        np_source = tests_cleared


    np_district = walk_through_permutations_for_clue(district, permutations_clues_matrix, 0)

    res = tuple(tuple(d) for d in np_district)
    return res

# clues = (3,2,2,3,2,1, 1,2,3,3,2,2, 5,1,2,2,4,3, 3,2,1,2,2,4)
clues = (0,0,0,2,2,0, 0,0,0,6,3,0, 0,4,0,0,0,0, 4,4,0,3,0,0)
# clues = (0,3,0,5,3,4, 0,0,0,0,0,1, 0,3,0,3,2,3, 3,2,0,3,1,0)

print(solve_puzzle(clues))










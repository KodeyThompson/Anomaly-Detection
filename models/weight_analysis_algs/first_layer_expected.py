import sys
import torch
import time

def reduce_dimension(weights1, weights2, weights3):
    reduced1 = weights1[0]
    reduced2 = weights2[0]
    reduced3 = weights3[0]
    for index in range(1, len(weights1)):
        reduced1 += weights1[index]
        reduced2 += weights2[index]
        reduced3 += weights3[index]
    return reduced1, reduced2, reduced3

if __name__ == "__main__":
    start_time = time.time()

    model1 = torch.load(sys.argv[1])
    model2 = torch.load(sys.argv[2])
    model3 = torch.load(sys.argv[3])
    all_weights1 = []
    all_weights2 = []
    all_weights3 = []
    for index, key in enumerate(model1.keys()):
        if index == 1:
            break
        weights1 = model1[key].tolist()
        weights2 = model2[key].tolist()
        weights3 = model3[key].tolist()
        for _ in range(max(0, len(model1[key].size()) - 1)):
            weights1, weights2, weights3 = reduce_dimension(weights1, weights2, weights3)
        if len(model1[key].size()) - 1 >= 0:
            all_weights1 += weights1
            all_weights2 += weights2
            all_weights3 += weights3

    expected_diffs = []
    for index in range(len(all_weights1)):
        expected_diffs.append(all_weights2[index] - all_weights1[index])

    actual_diffs = []
    for index in range(len(all_weights1)):
        actual_diffs.append(all_weights3[index] - all_weights2[index])

    diff_of_diffs = 0
    for index in range(len(all_weights1)):
        diff_of_diffs += abs(expected_diffs[index] - actual_diffs[index])

    end_time = time.time()

    if True:
        print(diff_of_diffs)
    else:
        print(f'The absolute difference is {diff_of_diffs} between the two models with a size of {len(all_weights1)} parameters')
        print(f'This is an average change of {diff_of_diffs / len(all_weights1)} for each weight')
        print(f'Took {end_time - start_time} seconds')

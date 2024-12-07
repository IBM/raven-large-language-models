#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import argparse
import json
import random
import sys

import numpy as np
from tqdm import tqdm

thismodule = sys.modules[__name__]

LOWVAL = 0


def random_candidates(target_val, target_idx, maxval):
    """
    Generate random candidate anwers
    """
    answer_candidates = np.random.randint(low=LOWVAL, high=maxval, size=8)
    # Find a replacement value
    replace_val = target_val
    while replace_val == target_val:
        replace_val = np.random.randint(low=LOWVAL, high=maxval)

    answer_candidates[answer_candidates == target_val] = replace_val
    answer_candidates[target_idx] = target_val
    return answer_candidates


def unbiased_candidates(context_panels, maxval, strategy="random"):
    """Unbiased candidate panels generation, inspired by
    Stratified Rule-Aware Network for Abstract Visual Reasoning, Sheng Hu, Yuqing Ma et al., AAAI, 2021.


    Args:
        context_panels (np.array): matrix of context panels (3, 8) with dimensions (attributes, panel values)
        maxval (int): maximum value of the attributes (only used with random strategy).
        strategy (str, optional): strategy used to generate the random attribute value in the branching
            operation of the binary tree. Defaults to "random". Possible values:
                - "random": the branching value is chosen at random in the interval [0, maxval[
                - "existent": the b. value is chosen within one of the values of any attribute in the context panels
                - "existent_att": the b. value is chosen within one of the values of the same attribute
                                  in the context panels

    Raises:
        ValueError: if strategy is not in ["random", "existent", "existent_att"]

    Returns:
        (np.array, int): returns the list of candidate panels (3, 8) and the position of the answer panel in that list.
    """
    answer = context_panels[:, -1, -1]
    wvals = list()
    for i in range(3):
        wval = answer[i]
        while wval == answer[i]:
            if strategy == "random":
                wval = random.randint(low=LOWVAL, high=maxval)
            elif strategy == "existent":
                wval = np.random.choice(np.unique(context_panels))
            elif strategy == "existent_att":
                wval = np.random.choice(np.unique(context_panels[i]))
            else:
                raise ValueError("Strategy not implemented")

        wvals.append(wval)

    def recursive_tweak(i, panel):
        if i == 3:
            return panel[None, :]
        tweaked = np.copy(panel)
        tweaked[i] = wvals[i]
        return np.concatenate(
            (recursive_tweak(i + 1, panel), recursive_tweak(i + 1, tweaked)), axis=0
        )

    candidates = np.random.permutation(recursive_tweak(0, answer))
    target = np.where((candidates == answer).all(axis=1))[0][0]
    return candidates, target


def Constant(n, maxval=50, *kwargs):
    """
    Generate context matrix for constant rule
    """
    context = np.tile(
        np.random.randint(low=LOWVAL, high=maxval, size=n), (n, 1)
    ).transpose()
    return context


def Progression(n, maxval=50, *kwargs):
    """
    Generate context matrix for progression rule
    Progression delta can be within {-2, -1, 1, 2}
    """
    delta = random.choice([-2, -1, 1, 2])

    context = np.zeros(
        (n, n),
    )

    # progression + (start from the left)
    if delta > 0:
        context[:, 0] = np.random.randint(
            low=LOWVAL, high=maxval - (n - 1) * delta, size=n
        )
        for col in range(1, n):
            context[:, col] = context[:, col - 1] + delta

    # progression - (start from the right)
    elif delta < 0:
        # start from right side, delta is negative, hence we add it for range computation
        context[:, -1] = np.random.randint(
            low=LOWVAL, high=maxval + ((n - 1) * delta), size=n
        )

        for col in range(n - 2, -1, -1):
            context[:, col] = context[:, col + 1] - delta
    return context


def Distribute_Three(n, maxval=50, *kwargs):
    """
    Generate context matrix for distribute-three rule
    For n>3, the rule is interpreted as distribute-n with cyclic permutations
    """
    context = np.zeros((n, n))
    arr = np.arange(LOWVAL, maxval)
    np.random.shuffle(arr)
    context[0, :] = arr[:n]
    delta = random.choice([-1, 1])

    for row in range(1, n):
        context[row, :] = np.roll(context[row - 1, :], delta)
    return context


def Arithmetic(n, maxval=50, arithmetic_strategy="shuffle"):
    """
    Generate context matrix for arithmetic rule
    Integrated two arithmetic strategies:
        uniform: sample values from maxval/(n-1) such that sum is <maxval
        shuffle: iterative sampling, where the range is decremented by current sum
    """

    sign = random.choice([-1, 1])  # decide between arithmetic plus and minus
    context = np.zeros((n, n))

    if arithmetic_strategy == "uniform":
        context_sum_operands = np.random.randint(
            low=LOWVAL, high=maxval / (n - 1), size=(n, n - 1)
        )
    elif arithmetic_strategy == "shuffle":
        context_sum_operands = generate_arithmetic_shuffle(n, maxval)

    # arithmetic +
    if sign > 0:
        context[:, :-1] = context_sum_operands
        context[:, -1] = np.sum(context, axis=1)
    # arithmetic -
    elif sign < 0:
        context[:, 1:] = context_sum_operands
        context[:, 0] = np.sum(context, axis=1)
    return context


def generate_arithmetic_shuffle(n, maxval):
    """
    Iterative sampling for generating context of arithmetic
    The range is decremented by current sum
    """
    context = np.zeros((n, n - 1))

    for row in range(n):
        target_sum = np.random.randint(low=maxval / 2, high=maxval)
        curr_maxval = target_sum + 1
        for col in range(n - 1):
            context[row, col] = np.random.randint(low=LOWVAL, high=curr_maxval)
            curr_maxval = target_sum - np.sum(context[row, :]) + 1

    # randomly shuffle along the rows
    rng = np.random.default_rng()
    context_permuted = rng.permuted(context, axis=1)

    # make sure that left-most value is no LOWVAL
    for row in range(n):
        if context_permuted[row, -1] == LOWVAL:
            non_zero_idx = np.nonzero(context_permuted[row, :] - LOWVAL)[0]
            context_permuted[row, -1] = context_permuted[row, non_zero_idx[0]]
            context_permuted[row, non_zero_idx[0]] = LOWVAL

    return context_permuted


def set_seeds(seed: int):
    """
    Set all the seeds for reproducible results
    """
    np.random.seed(seed)
    random.seed(seed)


def get_sample(n, maxval, rule, arithmetic_strategy):
    rpm = []

    # sample the rules
    rules = {}
    rules["Number/Position"] = "Constant"
    if rule == "":
        rules["Type"] = random.choice(["Constant", "Progression", "Distribute_Three"])
        rules["Size"] = random.choice(
            ["Constant", "Progression", "Distribute_Three", "Arithmetic"]
        )
        rules["Color"] = random.choice(
            ["Constant", "Progression", "Distribute_Three", "Arithmetic"]
        )
    else:
        rules["Type"] = rule
        rules["Size"] = rule
        rules["Color"] = rule

    type_val = getattr(thismodule, rules["Type"])(n, maxval, arithmetic_strategy)

    size_val = getattr(thismodule, rules["Size"])(n, maxval, arithmetic_strategy)

    color_val = getattr(thismodule, rules["Color"])(n, maxval, arithmetic_strategy)

    candidates, target = unbiased_candidates(
        np.stack((type_val, size_val, color_val)), maxval, strategy="existent"
    )

    type_val = np.concatenate([type_val.flatten()[:-1], candidates[:, 0]])
    color_val = np.concatenate([color_val.flatten()[:-1], candidates[:, 2]])

    # decline size value by 1 since it is incremented by 1 in the dataloader
    # (to be in line with the I-RAVEN representation)
    size_val = np.concatenate([size_val.flatten()[:-1], candidates[:, 1]]) - 1

    rpm = []
    for panel in range(n**2 + 7):
        ent_dict = {}
        ent_dict["Type"] = str(int(type_val[panel]))
        ent_dict["Size"] = str(int(size_val[panel]))
        ent_dict["Color"] = str(int(color_val[panel]))
        ent_dict["Angle"] = str(int(np.random.randint(low=LOWVAL, high=10)))
        rpm.append([ent_dict])

    return {"rules": [rules], "rpm": rpm, "target": target}


def generate(n, maxval, save_dir, rule, arithmetic_strategy):
    samples = {}
    for i in tqdm(range(10000)):
        samples[i] = get_sample(n, maxval, rule, arithmetic_strategy)

    save_fn = "{}/center_single{}_{}_n_{}_maxval_{}.json".format(
        save_dir, rule, arithmetic_strategy, n, maxval
    )
    json.dump(samples, open(save_fn, "w"), indent=1, default=str)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--maxval", type=int)
    parser.add_argument("--save_dir")
    parser.add_argument("--rule", type=str, default="")
    parser.add_argument("--arithmetic_strategy", type=str, default="shuffle")
    args = parser.parse_args()

    set_seeds(1234)
    generate(args.n, args.maxval, args.save_dir, args.rule, args.arithmetic_strategy)
    return


if __name__ == "__main__":
    main()

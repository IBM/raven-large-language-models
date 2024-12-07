#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import json

import numpy as np
from tqdm import tqdm

from .datasets.rpm_dataset import RPM
from .models import get_model


class Solver:
    def __init__(
        self,
        cfg,
        **kwargs,
    ):
        self.output = {}
        self.cfg = cfg
        self.model = get_model(self.cfg, **kwargs)
        return

    def __call__(self):
        """
        Main testing procedure:
        - Data loading
        - interaction w/ LLM
        - evaluation
        """

        with open(
            "{}/{}.json".format(self.cfg.data.path, self.cfg.data.config), "r"
        ) as f:
            samples = json.load(f)

        acc = np.zeros((2))

        if self.cfg.data.dataset == "iraven" or self.cfg.data.dataset == "iravenx":
            subset = np.random.permutation(np.arange(8000, 10000))  #
            incontext_subset = np.random.permutation(np.arange(6000))

        elif self.cfg.data.dataset == "ravenvariations":
            subset = [int(i) for i in samples.keys()]
            incontext_subset = subset

        # OOD filtering
        if self.cfg.data.ood_set_attr:
            raise ValueError("OOD not implemented yet")

        pbar = tqdm(subset[: self.cfg.data.ntest])
        for j, i in enumerate(pbar):

            test_sample = samples[str(i)]
            incontext_samples = self._getincontext(
                test_sample, samples, np.random.permutation(incontext_subset)
            )

            # Interaction with LLM
            if self.cfg.model.disentangled:  # do split predictions
                interaction, choices = self._split(
                    test_sample,
                    self.cfg.data.config,
                    incontext_samples,
                    n=self.cfg.data.gridsize,
                    nshow=self.cfg.data.nshow,
                    add_angle=self.cfg.data.angle,
                )
            else:  # do only coupled prediction (master)
                interaction, choices = self._merge(
                    test_sample,
                    self.cfg.data.config,
                    incontext_samples,
                    n=self.cfg.data.gridsize,
                    nshow=self.cfg.data.nshow,
                    add_angle=self.cfg.data.angle,
                )

            # Determine the predicted panel based on text interaction
            pred_sol, pred_attr = self._predict(
                interaction, self.cfg.model.disentangled, choice_array=choices
            )

            corr = self._get_correct(pred_sol, test_sample["target"])

            acc += np.array(corr)
            acc_print = acc / (j + 1)
            self.output[str(i)] = {
                "interaction": interaction,
                "correct": corr,
                "pred_sol": pred_sol,
                "pred_attr": pred_attr,
                "rules": test_sample["rules"],
                "choices": choices.tolist(),
                "target": test_sample["target"],
            }

            pbar.set_description(
                "Merge {:.3f} Split {:.3f}".format(acc_print[0], acc_print[1])
            )

        self.output["acc_merge"] = acc[0] / self.cfg.data.ntest
        self.output["acc_split"] = acc[1] / self.cfg.data.ntest

        # Dump everything to JSON file
        file_name = "{:}/result.json".format(self.cfg.path.full)
        if self.cfg.model.name != "null":
            json.dump(self.output, open(file_name, "w"), indent=1)
            self.output, self.context = {}, None
        return

    def _split():
        pass

    def _merge():
        pass

    def _predict():
        pass

    def _getincontext(self, test_sample, icl_candidate_samples, idx_set):
        """
        Extract a set of in-context samples, removing ones that overlap
        with the actual task.
        """
        out = []
        ctr = 0
        while len(out) < self.cfg.model.incontext and ctr < len(icl_candidate_samples):
            candidate = icl_candidate_samples[str(idx_set[ctr])]
            if not self._overlap_attr_test(candidate, test_sample):
                out.append(candidate)
            ctr += 1
        return out

    def _overlap_attr_test(self, x1, x2):
        """
        Checks if one of the attributes has equal
        constellation matrix. If yes, it returns True.
        """
        x1_rpm = RPM(
            x1,
            self.cfg.data.config,
            n=self.cfg.data.gridsize,
            nshow=self.cfg.data.nshow,
            add_angle=False,
        )
        x2_rpm = RPM(
            x2,
            self.cfg.data.config,
            n=self.cfg.data.gridsize,
            nshow=self.cfg.data.nshow,
            add_angle=False,
        )

        for i, component in enumerate(x1_rpm.components):
            for j, branch in component.branches.items():
                if x2_rpm.components[i].branches[j].context == branch.context:
                    return True
        return False

    def _get_correct(self, pred, sol):
        return int(int(sol) == int(pred[0])), int(int(sol) == int(pred[1]))

    def _filter_ood(self, samples, subset, ood_set_attr, ood_set_rule):
        """
        OOD filtering -> not used ATM
        """
        for idx in subset:
            if samples[str(idx)]["rules"][0][ood_set_attr] != ood_set_rule:
                subset = np.delete(subset, np.where(subset == idx))

        return subset, subset.shape[0]

    def _filter_ood_train(self, samples, subset, ood_set_attr, ood_set_rule):
        """
        OOD training data filtering -> not used ATM
        """
        train_attr = ["Size", "Color", "Type"]
        train_attr.remove(ood_set_attr)
        out_subset = []
        for tr_attr in train_attr:
            for idx in subset:
                if (samples[str(idx)]["rules"][0][ood_set_attr] != ood_set_rule) and (
                    samples[str(idx)]["rules"][0][tr_attr] == ood_set_rule
                ):
                    out_subset.append(idx)
        return np.array(out_subset)

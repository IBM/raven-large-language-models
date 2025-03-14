#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
import json
import os

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

        ####### Catch all the exceptions for wrong settings #############
        if self.cfg.data.ood_set_attr:
            raise ValueError("OOD not implemented yet")

        # General check for center single
        if "center_single" not in self.cfg.data.config:
            if self.cfg.model.classmode == "pred":
                raise ValueError(
                    "Predictive mode not suported for larger grids beyond center"
                )
            if self.cfg.model.disentangled:
                raise ValueError(
                    "Disentangled operation not implemented for larger grids beyond center"
                )

        # Confounders check
        if (
            self.cfg.model.disentangled or self.cfg.model.classmode == "pred"
        ) and self.cfg.data.nconf > 1:
            raise ValueError("Confounders are only for entangled discriminative mode")

        # Uncertainty check
        if (
            self.cfg.model.disentangled or self.cfg.model.classmode == "pred"
        ) and self.cfg.data.uncertainty is not None:
            raise ValueError("Uncertainty is only for disentangled discriminative mode")
        ##################################################################################

        # Load dataset
        with open(
            "{}/{}.json".format(self.cfg.data.path, self.cfg.data.config), "r"
        ) as f:
            samples = json.load(f)

        acc = np.zeros((2))

        subset = np.random.permutation(np.arange(8000, 10000))  # Test set
        incontext_subset = np.random.permutation(
            np.arange(6000)
        )  # Training set (for incontext examples)

        # Load file for saving results (if already partly done)
        file_name = "{:}/result.json".format(self.cfg.path.full)
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                self.output = json.load(f)
        else:
            print("No previous results loaded")

        pbar = tqdm(subset[: self.cfg.data.ntest])
        for j, i in enumerate(pbar):
            newprediction = True
            if str(i) in self.output:
                if self.output[str(i)]["interaction"]["out"][0] != "empty":
                    newprediction = False
                    acc += np.array(self.output[str(i)]["correct"])
                    acc_print = acc / (j + 1)

            if newprediction:  # make a new predcition
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
                        offset=self.cfg.data.offset,
                        scaling=self.cfg.data.scaling,
                    )
                else:  # do only coupled prediction (master)
                    interaction, choices = self._merge(
                        test_sample,
                        self.cfg.data.config,
                        incontext_samples,
                        n=self.cfg.data.gridsize,
                        nshow=self.cfg.data.nshow,
                        add_angle=self.cfg.data.angle,
                        offset=self.cfg.data.offset,
                        scaling=self.cfg.data.scaling,
                        nconf=self.cfg.data.nconf,
                        permutation=self.cfg.data.permutation,
                        uncertainty=self.cfg.data.uncertainty,
                        maxval_uncert=self.cfg.data.maxval_uncert,
                    )

                # Determine the predicted panel based on text interaction
                pred_sol, pred_attr = self._predict(
                    interaction,
                    self.cfg.model.disentangled,
                    choice_array=choices,
                    answer_queue=self.cfg.model.answer_queue,
                    scaling=self.cfg.data.scaling,
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

                # print(interaction)

            pbar.set_description(
                "Merge {:.3f} Split {:.3f}".format(acc_print[0], acc_print[1])
            )

            # Intermediate dump everything to JSON file
            if self.cfg.model.name != "null":
                json.dump(self.output, open(file_name, "w"), indent=1)

        # Final accuracy computation
        self.output["acc_merge"] = acc[0] / self.cfg.data.ntest
        self.output["acc_split"] = acc[1] / self.cfg.data.ntest

        # Final dump to JSON file
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

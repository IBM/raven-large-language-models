#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import re

import numpy as np
from scipy.stats import mode

from .datasets.rpm_dataset import RPM
from .solver import Solver


def find_best_match(sol, pred) -> int:
    """
    Select the best matching panel
    """
    correct = ((sol - pred) == 0).astype(int)
    n_correct = correct.sum(axis=-1)
    return int(np.argmax(n_correct))


def majority_vote(pred_array) -> np.array:
    """
    Majority vote, mainly used in self-consistency configurations
    """
    maj, _ = mode(pred_array, axis=0, nan_policy="omit")
    maj[np.isnan(maj)] = 0
    return np.squeeze(maj.data)


def text2num(text, nattr, nreturn) -> np.array:
    """
    Extract the predicted attributes from textual LLM output
    """
    pred_array = np.zeros((nreturn, nattr))
    pred_array[:] = np.nan
    for i, pred_text in enumerate(text):
        pred_num = np.array(
            [float(s) for s in re.findall(r"[-+]?(?:\d*\.*\d+)", pred_text)]
        )
        if pred_num.shape[0] >= nattr:
            pred_array[i, :] = pred_num[:nattr]
    return pred_array


def scale_rep(x, scale=[1, 10, 0.1]):
    """
    Scale the attributes to a specific range
    """
    for i in range(len(scale)):
        x[:, i] = x[:, i] * scale[i]
    return x


class Solver_pred(Solver):
    """
    Predictive solver
    """

    def __init__(self, cfg, **kwargs):
        super(Solver_pred, self).__init__(cfg, **kwargs)
        self.prompt = cfg.model.prompt
        self.prefix = cfg.model.prefix

    def _split(self, sample, config, incontext_samples, n=3, nshow=3, add_angle=False):
        """
        Disentangled classification using an LLM
        In addition, the entangled classification (master) is performed as well.
        """

        ret = []
        # initialize the RPM sample
        rpm = RPM(sample, config, n=n, nshow=nshow, add_angle=add_angle)
        for i, component in enumerate(rpm.components):

            ret.append({})
            for j, branch in component.branches.items():

                # extract incontext examples
                incontext = ""
                for ex_idx, sa in enumerate(incontext_samples):
                    ic_rpm = RPM(sa, config, n=n, nshow=nshow, add_angle=add_angle)
                    incontext += (
                        "Example {:}:\n".format(ex_idx)
                        + ic_rpm.components[i].branches[j].context
                        + ic_rpm.components[i].branches[j].choices[sa["target"]]
                        + ";\n\n"
                    )
                if incontext != "":
                    incontext += "Your question:\n"

                query = self.prefix + incontext + branch.context

                # Perform entangled querying as well
                if j == "master":
                    query += "("

                interaction = self.model.forward(self.prompt, query)

                ret[i][j] = interaction

            # Scale representations for later comparison
            choices = scale_rep(text2num(rpm.choices, self.cfg.data.nattr, 8))

        return ret, choices

    def _merge(self, sample, config, incontext_samples, n=3, nshow=3, add_angle=False):
        """
        Perform entangled classification only
        """
        rpm = RPM(sample, config, n=n, nshow=nshow, add_angle=add_angle)

        # Extract incontext info
        incontext = ""
        for ex_idx, sa in enumerate(incontext_samples):
            ic_rpm = RPM(sa, config, n=n, nshow=nshow, add_angle=add_angle)
            incontext += (
                "Example {:}:\n".format(ex_idx)
                + ic_rpm.context
                + ic_rpm.choices[sa["target"]]
                + ";\n\n"
            )
        if incontext != "":
            incontext += "Your question:\n"

        query = self.prefix + incontext + rpm.context + "("

        scores = self.model.forward(self.prompt, query)

        # Scale representations for later comparison
        choices = scale_rep(text2num(rpm.choices, self.cfg.data.nattr, 8))

        return scores, choices

    def _predict(self, x, disentangled, choice_array):
        """
        Predict the attributes based on textual output of LLM (x)
        """
        if disentangled:  # for split
            ensemble_pred = np.zeros((self.cfg.model.nreturn, 0))
            for comp_dict in x:
                for branch_name, branch_dict in comp_dict.items():
                    if branch_name == "master":
                        master_pred = scale_rep(
                            text2num(
                                branch_dict["out"],
                                self.cfg.data.nattr,
                                self.cfg.model.nreturn,
                            )
                        )
                    else:
                        ensemble_pred = np.append(
                            ensemble_pred,
                            text2num(branch_dict["out"], 1, self.cfg.model.nreturn),
                            axis=-1,
                        )
            # master prediction (entangled)
            master_pred_maj = majority_vote(master_pred)
            master_pred_sol = find_best_match(choice_array, master_pred_maj)

            # ensemble predicition (disentangled)
            ensemble_pred = np.array(ensemble_pred)
            ensemble_pred_maj = majority_vote(ensemble_pred)
            ensemble_pred_sol = find_best_match(choice_array, ensemble_pred_maj)

            return [master_pred_sol, ensemble_pred_sol], [
                master_pred_maj.tolist(),
                ensemble_pred_maj.tolist(),
            ]
        else:
            # Only entangled prediction
            pred_array = scale_rep(
                text2num(x, self.cfg.data.nattr, self.cfg.model.nreturn)
            )
            pred_array_maj = majority_vote(pred_array)
            pred_sol = find_best_match(choice_array, pred_array_maj)

            return [pred_sol, 0], [pred_array_maj.tolist(), 0]

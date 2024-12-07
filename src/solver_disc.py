#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np

from .datasets.rpm_dataset import RPM
from .solver import Solver
from .solver_pred import find_best_match, majority_vote, scale_rep, text2num


def choice2val(prediction, choices):
    n_pred, n_attr = prediction.shape
    out = np.zeros((n_pred, n_attr))
    for pred in range(n_pred):
        for attr in range(n_attr):
            out[pred, attr] = choices[int(prediction[pred, attr]), attr]
    return out


class Solver_disc(Solver):
    def __init__(self, cfg, **kwargs):
        """
        Implement discriminative solver by providing solutions to the
        language model.
        """
        super(Solver_disc, self).__init__(cfg, **kwargs)
        self.prompt = cfg.model.prompt
        self.prefix = cfg.model.prefix

    def _split(self, sample, config, incontext_samples, n=3, add_angle=False):
        ret = []
        rpm = RPM(sample, config, n=n, add_angle=add_angle)
        for i, component in enumerate(rpm.components):

            ret.append({})
            for j, branch in component.branches.items():

                # extract incontext examples
                incontext = ""
                for sa in incontext_samples:
                    ic_rpm = RPM(sa, config, n=n, add_angle=add_angle)
                    incontext += (
                        ic_rpm.components[i].branches[j].context
                        + ic_rpm.components[i].branches[j].choices[sa["target"]]
                        + ";\n\n"
                    )

                query = (
                    self.prefix
                    + incontext
                    + branch.context
                    + "\nSelect the correct Answer from the following list\n"
                    + branch.answer_choices
                    + "\nSolution: The correct answer is Answer #"
                )

                interaction = self.model.forward(self.prompt, query)

                ret[i][j] = interaction

            choices = scale_rep(text2num(rpm.choices, self.cfg.data.nattr, 8))
        return ret, choices

    def _merge(self, sample, config, incontext_samples, n=3, add_angle=False):
        rpm = RPM(sample, config, n=n, add_angle=add_angle)

        # Extract incontext info
        incontext = ""
        for sa in incontext_samples:
            ic_rpm = RPM(sa, config, n=n, add_angle=add_angle)
            incontext += ic_rpm.context + ic_rpm.choices[sa["target"]] + ";\n\n"

        query = (
            self.prefix
            + incontext
            + rpm.context
            + "\nSelect the correct Answer from the following list\n"
            + rpm.answer_choices
            + "\nCorrect Answer #"
        )

        interaction = self.model.forward(self.prompt, query)

        choices = scale_rep(text2num(rpm.choices, self.cfg.data.nattr, 8))
        return interaction, choices

    def _predict(self, x, disentangled, choice_array):

        if disentangled:  # for split
            ensemble_pred = np.zeros((self.cfg.model.nreturn, 0))
            for comp_dict in x:
                for branch_name, branch_dict in comp_dict.items():
                    if branch_name == "master":
                        master_pred = text2num(
                            branch_dict["out"],
                            1,  # we only extract the answer panel id, not the attributes. Hence only 1 instead of nattr
                            self.cfg.model.nreturn,
                        )
                    else:
                        ensemble_pred = np.append(
                            ensemble_pred,
                            text2num(branch_dict["out"], 1, self.cfg.model.nreturn),
                            axis=-1,
                        )

            # master prediction
            master_pred_sol = int(majority_vote(master_pred))
            master_pred_val = choice_array[master_pred_sol]

            # ensemble predicition
            ensemble_pred = np.array(ensemble_pred)
            ensemble_val = choice2val(ensemble_pred, choice_array)
            ensemble_pred_val = majority_vote(ensemble_val)
            ensemble_pred_sol = find_best_match(choice_array, ensemble_pred_val)

            return [master_pred_sol, ensemble_pred_sol], [
                master_pred_val.tolist(),
                ensemble_pred_val.tolist(),
            ]
        else:
            master_pred = text2num(x["out"], 1, self.cfg.model.nreturn)
            master_pred_sol = int(majority_vote(master_pred))
            master_pred_val = choice_array[master_pred_sol]

            return [master_pred_sol, 0], [master_pred_val.tolist(), 0]

#
# Copyright 2025- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np

from .datasets.rpm_dataset import RPM
from .solver import Solver
from .solver_pred import find_best_match, majority_vote, scale_rep, text2num


def choice2val(prediction, choices):
    n_pred, n_attr = prediction.shape
    out = np.zeros((n_pred, n_attr))
    prediction[np.isnan(prediction)] = 0  # set default to 0 panel if nan
    for pred in range(n_pred):
        for attr in range(n_attr):
            my_idx = int(prediction[pred, attr])
            # map answer id to attribute
            out[pred, attr] = choices[my_idx, attr]
    return out


def guard_answer(x):
    """
    Sets values in array between 0 and 7.
    Out of bound values are set to 0
    """
    x[x < 0] = 0
    x[x > 7] = 0
    return x


class Solver_disc(Solver):
    def __init__(self, cfg, **kwargs):
        """
        Implement discriminative solver by providing solutions to the
        language model.
        """
        super(Solver_disc, self).__init__(cfg, **kwargs)
        self.prompt = cfg.model.prompt
        self.prefix = cfg.model.prefix

    def _split(
        self,
        sample,
        config,
        incontext_samples,
        n=3,
        nshow=3,
        add_angle=False,
        offset=True,
        scaling=True,
    ):
        ret = []
        rpm = RPM(
            sample,
            config,
            n=n,
            nshow=nshow,
            add_angle=add_angle,
            offset=offset,
            scaling=scaling,
        )
        for i, component in enumerate(rpm.components):

            ret.append({})
            for j, branch in component.branches.items():

                # extract incontext examples
                incontext = ""
                for sa_idx, sa in enumerate(incontext_samples):
                    ic_rpm = RPM(
                        sa,
                        config,
                        n=n,
                        nshow=nshow,
                        add_angle=add_angle,
                        offset=offset,
                        scaling=scaling,
                    )
                    incontext += "Example {}:\n{}\nAnswer set:\n{}\nMy Answer: Answer #{}\n\n".format(
                        sa_idx, ic_rpm.context, ic_rpm.answer_choices, sa["target"]
                    )
                query = (
                    self.prefix
                    + incontext
                    + branch.context
                    + "\nAnswer set:\n"
                    + branch.answer_choices
                )

                interaction = self.model.forward(self.prompt, query)

                ret[i][j] = interaction

            # extract choice values on master
            choices = text2num(rpm.choices, self.cfg.data.nattr, 8)
            if scaling:
                choices = scale_rep(choices)
        return ret, choices

    def _merge(
        self,
        sample,
        config,
        incontext_samples,
        n=3,
        nshow=3,
        add_angle=False,
        offset=True,
        scaling=True,
        nconf=0,
        permutation=None,
        uncertainty=None,
        maxval_uncert=None,
    ):
        rpm = RPM(
            sample,
            config,
            n=n,
            nshow=nshow,
            add_angle=add_angle,
            offset=offset,
            scaling=scaling,
            nconf=nconf,
            permutation=permutation,
            uncertainty=uncertainty,
            maxval_uncert=maxval_uncert,
        )

        # Extract incontext info
        incontext = ""
        for sa_idx, sa in enumerate(incontext_samples):
            ic_rpm = RPM(
                sa,
                config,
                n=n,
                nshow=nshow,
                add_angle=add_angle,
                offset=offset,
                scaling=scaling,
                nconf=nconf,
                permutation=permutation,
                uncertainty=uncertainty,
                maxval_uncert=maxval_uncert,
            )
            incontext += (
                "Example {}:\n{}\nAnswer set:\n{}\nMy Answer: Answer #{}\n\n".format(
                    sa_idx, ic_rpm.context, ic_rpm.answer_choices, sa["target"]
                )
            )

        query = (
            self.prefix
            + incontext
            + rpm.context
            + "\nAnswer set:\n"
            + rpm.answer_choices
        )

        interaction = self.model.forward(self.prompt, query)
        # extract anser candidates
        choices = text2num(rpm.choices, self.cfg.data.nattr + self.cfg.data.nconf, 8)
        if scaling:
            choices = scale_rep(choices)
        return interaction, choices

    def _predict(self, x, disentangled, choice_array, answer_queue="", **kwargs):

        if disentangled:  # for split
            ensemble_pred = np.zeros((self.cfg.model.nreturn, 0))
            for comp_dict in x:
                for branch_name, branch_dict in comp_dict.items():
                    if branch_name == "master":
                        master_pred = text2num(
                            branch_dict["out"],
                            1,  # we only extract the answer panel id, not the attributes. Hence only 1 instead of nattr
                            self.cfg.model.nreturn,
                            answer_queue,
                        )
                    else:
                        ensemble_pred = np.append(
                            ensemble_pred,
                            text2num(
                                branch_dict["out"],
                                1,
                                self.cfg.model.nreturn,
                                answer_queue,
                            ),
                            axis=-1,
                        )

            # master prediction
            master_pred = guard_answer(master_pred)
            master_pred_sol = int(majority_vote(master_pred))
            master_pred_val = choice_array[master_pred_sol]

            # ensemble predicition
            ensemble_pred = np.array(ensemble_pred)
            ensemble_pred = guard_answer(ensemble_pred)
            ensemble_val = choice2val(ensemble_pred, choice_array)
            ensemble_pred_val = majority_vote(ensemble_val)
            ensemble_pred_sol = find_best_match(choice_array, ensemble_pred_val)

            return [master_pred_sol, ensemble_pred_sol], [
                master_pred_val.tolist(),
                ensemble_pred_val.tolist(),
            ]
        else:
            # master prediction on panel ID directly
            master_pred = text2num(x["out"], 1, self.cfg.model.nreturn, answer_queue)
            master_pred = guard_answer(master_pred)
            master_pred_sol = int(majority_vote(master_pred))

            master_pred_val = choice_array[master_pred_sol]

            return [master_pred_sol, 0], [master_pred_val.tolist(), 0]

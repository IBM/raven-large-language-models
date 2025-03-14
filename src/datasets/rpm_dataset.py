#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import math

import numpy as np


class Shape:
    def __init__(
        self,
        shape_dict,
        add_angle=False,
        offset=True,
        scaling=True,
        nconf=0,
        permutation=None,
        uncertainty=None,
        maxval_uncert=None,
    ):
        if offset:
            self.type_offset = 2
            self.size_offset = 1
        else:
            self.type_offset = 0
            self.size_offset = 0

        if scaling:  # this is the scaling proposed by Hu et al. ACL 2023
            self.type_scale = 1
            self.size_invscale = (
                10  # this inverse scaling (1/x). Prevents floating point errors
            )
            self.color_scale = 10
            self.angle_scale = 100
        else:  # no scaling
            self.type_scale = 1
            self.size_invscale = 1
            self.color_scale = 1
            self.angle_scale = 1

        self.type = self._type(shape_dict["Type"])
        self.size = self._size(shape_dict["Size"])
        self.color = self._color(shape_dict["Color"])
        # Angle generation
        if add_angle:
            self.angle = self._angle(shape_dict["Angle"])
        self.add_angle = add_angle
        # General confounder generation
        self.nconf = nconf
        for i in range(self.nconf):
            setattr(
                self,
                "confounder" + str(i),
                self._angle(shape_dict["Confounder" + str(i)]),
            )
        self.permutation = permutation

        if uncertainty is not None:
            self._conv = self._uncertainty_conv
        else:
            self._conv = self._no_conv

        self.uncertainty = uncertainty
        self.maxval_uncert = maxval_uncert

    def _type(self, x):
        return int(x) + self.type_offset

    def _size(self, x):
        return int(x) + self.size_offset

    def _color(self, x):
        return int(x)

    def _angle(self, x):
        return int(x)

    def _confounder(self, x):
        return int(x)

    def _no_conv(self, x):
        return x

    def _uncertainty_conv(self, x):
        # sample probabilities
        p_x = round(np.random.uniform(low=self.uncertainty, high=1.0), 2)
        p_side = np.zeros(2)
        p_side[0] = round(np.random.uniform(low=0, high=1 - p_x), 2)
        p_side[1] = 1 - p_x - p_side[0]
        p_side = np.random.permutation(p_side)

        # determine left and right values values
        x_m1 = x - 1 if x > 0 else self.maxval_uncert - 1
        x_p1 = x + 1 if x < self.maxval_uncert - 1 else 0

        # write to string
        out_str = "<{:.2f}::{},{:.2f}::{},{:.2f}::{}>".format(
            p_side[0], x_m1, p_x, x, p_side[1], x_p1
        )
        return out_str

    def __str__(self):
        val_array = []

        val_array.append(self.type * self.type_scale)

        # this inverse scaling (1/x). Prevents floating point errors
        val_size = (
            self.size / self.size_invscale if self.size_invscale != 1 else self.size
        )
        val_array.append(val_size)

        val_array.append(self.color * self.color_scale)

        # Add angle if required
        if self.add_angle:
            val_array.append(self.angle * self.angle_scale)

        # Add confounders
        elif self.nconf > 0:
            for i in range(self.nconf):
                val_array.append(getattr(self, "confounder" + str(i)))

        if isinstance(self.permutation, list):
            val_array = np.array(val_array)[self.permutation]

        # Map value array to specific prompting format
        outstring = "({}".format(self._conv(val_array[0]))
        for i in range(1, len(val_array)):
            outstring = "{},{}".format(outstring, self._conv(val_array[i]))
        outstring = "{})".format(outstring)

        return outstring


class Grid:
    def __init__(self, grid_dict, dim, add_angle=False, offset=True, scaling=True):
        self.dim = dim
        self.add_angle = add_angle
        self.offset = offset
        self.scaling = scaling
        self.coords = self._coords(grid_dict["positions"])
        self.shapes = self._shapes(grid_dict["entities"])
        self.types, self.sizes, self.colors, self.angles = self._split()
        self.string = ""
        self.layout = []
        self._update()

    def _coords(self, coords):
        ret = []
        for coord in coords:
            x = int(math.ceil(coord[0] * self.dim))
            y = int(math.ceil(coord[1] * self.dim))
            ret.append((x, y))
        return ret

    def _shapes(self, shape_dicts):
        return [
            Shape(
                shape_dict,
                add_angle=self.add_angle,
                offset=self.offset,
                scaling=self.scaling,
            )
            for shape_dict in shape_dicts
        ]

    def _split(self):
        types, sizes, colors, angles = [], [], [], []
        for shape in self.shapes:
            types.append(shape.type)
            sizes.append(shape.size)
            colors.append(shape.color)
            if self.add_angle:
                angles.append(shape.angle)
        types = list(set(types))
        types.sort()
        sizes = list(set(sizes))
        sizes.sort()
        colors = list(set(colors))
        colors.sort()
        angles = list(set(angles))
        angles.sort()
        return types, sizes, colors, angles

    def _update(self):
        self.string += "["
        for i in range(self.dim**2):
            x = int(i / self.dim) + 1
            y = i % self.dim + 1
            if (x, y) in self.coords:
                j = self.coords.index((x, y))
                self.string += str(self.shapes[j])
                self.layout.append(1)
            else:
                self.string += "-"
                self.layout.append(0)
            if i < self.dim**2 - 1:
                self.string += ", "
        self.string += "]"
        return

    def __str__(self):
        return self.string

    def get_layout(self):
        return str(self.layout)

    def get_number(self):
        return sum(self.layout)

    def get_types(self):
        return str(self.types)

    def get_sizes(self):
        return str(self.sizes)

    def get_colors(self):
        return str(self.colors)

    def get_angles(self):
        return str(self.angles)


class Branch:
    def __init__(self, arr, n=3, nshow=3):
        self.context = self._context(arr, n, nshow)
        self.choices = [str(x) for x in arr[(n**2 - 1) :]]
        self.answer_choices = "\n".join(
            ["Answer #{}: {}".format(i, ans) for i, ans in enumerate(self.choices)]
        )

    def _context(self, arr, n, nshow):
        tpl = ""
        for row in range(nshow):
            tpl = tpl + "row " + str(row + 1) + ": {}"
            if row < nshow - 1:
                for _ in range(1, n):
                    tpl += ", {}"
                tpl += "; "
            else:
                for _ in range(1, n - 1):
                    tpl += ", {}"
                tpl += ", "

        return tpl.format(*arr[((n - nshow) * n) : (n**2 - 1)])


class Component:
    def __init__(
        self,
        item_dicts,
        config,
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
        self.config = config
        self.add_angle = add_angle
        self.nconf = nconf
        self.permutation = permutation
        self.uncertainty = uncertainty
        self.maxval_uncert = maxval_uncert
        self.offset = offset
        self.scaling = scaling
        self.items = self._items(item_dicts)
        self.branches = {}
        self._update(n, nshow)

    def _items(self, item_dicts):
        if self.config[:13] == "center_single":
            return [
                Shape(
                    item_dict,
                    add_angle=self.add_angle,
                    offset=self.offset,
                    scaling=self.scaling,
                    nconf=self.nconf,
                    permutation=self.permutation,
                    uncertainty=self.uncertainty,
                    maxval_uncert=self.maxval_uncert,
                )
                for item_dict in item_dicts
            ]
        elif self.config == "distribute_four":
            return [
                Grid(
                    item_dict,
                    2,
                    add_angle=self.add_angle,
                    offset=self.offset,
                    scaling=self.scaling,
                )
                for item_dict in item_dicts
            ]
        else:
            return [
                Grid(
                    item_dict,
                    3,
                    add_angle=self.add_angle,
                    offset=self.offset,
                    scaling=self.scaling,
                )
                for item_dict in item_dicts
            ]

    def _update(self, n, nshow):
        if self.config[:13] == "center_single":
            self.branches["type"] = [shape.type for shape in self.items]
            self.branches["size"] = [shape.size for shape in self.items]
            self.branches["color"] = [shape.color for shape in self.items]
            if self.add_angle:
                self.branches["angle"] = [shape.angle for shape in self.items]

            for i in range(self.nconf):
                self.branches["confounder" + str(i)] = [
                    getattr(shape, "confounder" + str(i)) for shape in self.items
                ]
        else:
            self.branches["type"] = [grid.get_types() for grid in self.items]
            self.branches["size"] = [grid.get_sizes() for grid in self.items]
            self.branches["color"] = [grid.get_colors() for grid in self.items]
            if self.add_angle:
                self.branches["angle"] = [grid.get_angles() for grid in self.items]

            self.branches["layout"] = [grid.get_layout() for grid in self.items]
            self.branches["number"] = [grid.get_number() for grid in self.items]
        self.branches["master"] = self.items
        for k in self.branches.keys():
            self.branches[k] = Branch(self.branches[k], n=n, nshow=nshow)


class RPM:
    def __init__(
        self,
        sample,
        config,
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
        """
        Generate textual prompts from RPM data

        Args:
        -----
        ------ General settings ---
        sample: RPM attribute data
        config: RPM constellation (e.g., "center_single")
        n:      RPM panel size (default 3)
        nshow:  Number of rows to be shown in textual prompts
        add_angle: Add angular attribute in textual prompts
        offset: Add offsets to some of the attributes (this is needed for I-RAVEN)
        scaling: Scale the attributes with a factor of 0.1, 1, and 10 in entangled prompt

        ------ I-RAVEN-X specific settings ---
        nconf: Number of confounders
        permuation: a list of integers describing the permutation in engangled prompt
        uncertainty: have a probability distribution instead of discrete values
        maxval_uncert: maximum attribute value in uncertainty distribution
        """

        self.config = config
        self.sample = sample
        self.offset = offset
        self.scaling = scaling
        self.add_angle = add_angle
        self.nconf = nconf
        self.permutation = permutation
        self.uncertainty = uncertainty
        self.maxval_uncert = maxval_uncert
        self.components = self._components(n, nshow)
        self.context = None
        self.choices = None
        self._update(n, nshow)

    def _components(self, n, nshow):
        item_dicts_0 = [self.sample["rpm"][j][0] for j in range(n**2 - 1 + 8)]
        if self.config[:13] == "center_single" or self.config[:10] == "distribute":
            return [
                Component(
                    item_dicts_0,
                    self.config,
                    n=n,
                    nshow=nshow,
                    add_angle=self.add_angle,
                    offset=self.offset,
                    scaling=self.scaling,
                    nconf=self.nconf,
                    permutation=self.permutation,
                    uncertainty=self.uncertainty,
                    maxval_uncert=self.maxval_uncert,
                )
            ]
        else:
            item_dicts_1 = [self.sample["rpm"][j][1] for j in range(n**2 - 1 + 8)]
            if self.config == "in_distribute_four_out_center_single":
                return [
                    Component(
                        item_dicts_0,
                        "center_single",
                        n=n,
                        nshow=nshow,
                        add_angle=self.add_angle,
                        offset=self.offset,
                        scaling=self.scaling,
                    ),
                    Component(
                        item_dicts_1,
                        "distribute_four",
                        n=n,
                        nshow=nshow,
                        add_angle=self.add_angle,
                        offset=self.offset,
                        scaling=self.scaling,
                    ),
                ]
            else:
                return [
                    Component(
                        item_dicts_0,
                        "center_single",
                        n=n,
                        nshow=nshow,
                        add_angle=self.add_angle,
                        offset=self.offset,
                        scaling=self.scaling,
                    ),
                    Component(
                        item_dicts_1,
                        "center_single",
                        n=n,
                        nshow=nshow,
                        add_angle=self.add_angle,
                        offset=self.offset,
                        scaling=self.scaling,
                    ),
                ]

    def _update(self, n, nshow):
        if self.config[:13] == "center_single" or self.config[:10] == "distribute":
            self.context = self.components[0].branches["master"].context
            self.choices = self.components[0].branches["master"].choices

        else:
            combined = []
            for x, y in zip(self.components[0].items, self.components[1].items):
                combined.append("A {} / B {}".format(x, y))
            if nshow == 3:
                tpl = "row 1: {}, {}, {}; row 2: {}, {}, {}; row 3: {}, {}, "
                self.context = tpl.format(*combined[: (n**2 - 1)])
            self.choices = combined[(n**2 - 1) :]

        self.answer_choices = "\n".join(
            ["Answer #{}: {}".format(i, ans) for i, ans in enumerate(self.choices)]
        )

        return

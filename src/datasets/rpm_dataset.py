#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import math


class Shape:
    def __init__(self, shape_dict, add_angle=False):
        self.type = self._type(shape_dict["Type"])
        self.size = self._size(shape_dict["Size"])
        self.color = self._color(shape_dict["Color"])
        if add_angle:
            self.angle = self._angle(shape_dict["Angle"])
        self.add_angle = add_angle

    def _type(self, x):
        return int(x) + 2

    def _size(self, x):
        return int(x) + 1

    def _color(self, x):
        return int(x)

    def _angle(self, x):
        return int(x)

    def __str__(self):
        if self.add_angle:
            return "({},{},{},{})".format(
                self.type, self.size / 10, self.color * 10, self.angle * 100
            )
        else:
            return "({},{},{})".format(self.type, self.size / 10, self.color * 10)


class Grid:
    def __init__(self, grid_dict, dim, add_angle=False):
        self.dim = dim
        self.add_angle = add_angle
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
            Shape(shape_dict, add_angle=self.add_angle) for shape_dict in shape_dicts
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
    def __init__(self, item_dicts, config, n=3, nshow=3, add_angle=False):
        self.config = config
        self.add_angle = add_angle
        self.items = self._items(item_dicts)
        self.branches = {}
        self._update(n, nshow)

    def _items(self, item_dicts):
        if self.config[:13] == "center_single":
            return [
                Shape(item_dict, add_angle=self.add_angle) for item_dict in item_dicts
            ]
        elif self.config == "distribute_four":
            return [
                Grid(item_dict, 2, add_angle=self.add_angle) for item_dict in item_dicts
            ]
        else:
            return [
                Grid(item_dict, 3, add_angle=self.add_angle) for item_dict in item_dicts
            ]

    def _update(self, n, nshow):
        if self.config[:13] == "center_single":
            self.branches["type"] = [shape.type for shape in self.items]
            self.branches["size"] = [shape.size for shape in self.items]
            self.branches["color"] = [shape.color for shape in self.items]
            if self.add_angle:
                self.branches["angle"] = [shape.angle for shape in self.items]
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
    def __init__(self, sample, config, n=3, nshow=3, add_angle=False):
        """
        Generate textual prompts from RPM data

        Args:
        -----
        sample: RPM attribute data
        config: RPM constellation (e.g., "center_single")
        n:      RPM panel size (default 3)
        nshow:  Number of rows to be shown in textual prompts
        add_angle: Add angular attribute in textual prompts
        """

        self.config = config
        self.sample = sample
        self.add_angle = add_angle
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
                    ),
                    Component(
                        item_dicts_1,
                        "distribute_four",
                        n=n,
                        nshow=nshow,
                        add_angle=self.add_angle,
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
                    ),
                    Component(
                        item_dicts_1,
                        "center_single",
                        n=n,
                        nshow=nshow,
                        add_angle=self.add_angle,
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

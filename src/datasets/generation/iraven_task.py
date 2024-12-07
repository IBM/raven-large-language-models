#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

"""
Extract visual attributes from I-RAVEN (or RAVEN) and dump it to JSON file
"""


def to_list(s):
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.split(",")
    return [float(n) for n in s]


def switch(arr, i, j):
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp
    return arr


def get_sample(file_name, target):
    tree = ET.parse(file_name)
    root = tree.getroot()
    rpm = []
    rules = []
    for child in root:
        if child.tag == "Panels":
            for panel in child:
                comps = []
                for struct in panel:
                    for comp in struct:
                        comp_dict = {}
                        for layout in comp:
                            if "Distribute" in layout.attrib["name"]:
                                comp_dict["positions"] = []
                                comp_dict["entities"] = []
                            for entity in layout:
                                ent_dict = {}
                                ent_dict["Type"] = entity.attrib["Type"]
                                ent_dict["Size"] = entity.attrib["Size"]
                                ent_dict["Color"] = entity.attrib["Color"]
                                ent_dict["Angle"] = entity.attrib["Angle"]
                                if "Distribute" in layout.attrib["name"]:
                                    pos = to_list(entity.attrib["bbox"])
                                    comp_dict["positions"].append(pos)
                                    comp_dict["entities"].append(ent_dict)
                                else:
                                    comp_dict = ent_dict
                        comps.append(comp_dict)
                rpm.append(comps)
        if child.tag == "Rules":
            for rule_group in child:
                comp_rules = {}
                for rule in rule_group:
                    attr = rule.attrib["attr"]
                    name = rule.attrib["name"]
                    comp_rules[attr] = name
                rules.append(comp_rules)
    # rpm = switch(rpm, 8, 8+target)
    return {"rules": rules, "rpm": rpm, "target": target}


def extract(config, load_dir, save_dir):
    samples = {}
    for i in tqdm(range(10000)):
        if i % 10 < 6:
            x = "train"
        elif i % 10 < 8:
            x = "val"
        else:
            x = "test"
        npz_fn = "{}/{}/RAVEN_{}_{}.npz".format(load_dir, config, i, x)
        xml_fn = "{}/{}/RAVEN_{}_{}.xml".format(load_dir, config, i, x)
        target = int(np.load(npz_fn)["target"])
        samples[i] = get_sample(xml_fn, target)
        # import pdb; pdb.set_trace()
    save_fn = "{}/{}.json".format(save_dir, config)
    json.dump(samples, open(save_fn, "w"), indent=1)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--load_dir")
    parser.add_argument("--save_dir")
    args = parser.parse_args()
    extract(args.config, args.load_dir, args.save_dir)
    return


if __name__ == "__main__":
    main()

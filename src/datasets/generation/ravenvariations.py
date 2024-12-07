#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import argparse
import glob
import json
import os
import re

"""
Extract visual attributes from Raven variations and dump it to JSON file
"""


def to_list(s):
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.split(",")
    return [float(n) for n in s]


def get_sample(data):

    rpm = []
    rules = []

    for panel in data["panels"]:
        comps = []
        for struct in panel:
            if "comp" in struct:
                comp_dict = {}
                if "Distribute" in panel["structure"]:
                    comp_dict["positions"] = []
                    comp_dict["entities"] = []
                for entity in panel[struct]:
                    ent_dict = {}
                    ent_dict["Type"] = str(entity["type"])
                    ent_dict["Size"] = str(entity["size"])
                    ent_dict["Color"] = str(entity["color"])
                    ent_dict["Angle"] = str(entity["angle"])
                    if "Distribute" in panel["structure"]:
                        pos = to_list(entity.attrib["bbox"])
                        comp_dict["positions"].append(pos)
                        comp_dict["entities"].append(ent_dict)
                    else:
                        comp_dict = ent_dict
                    comps.append(comp_dict)
        rpm.append(comps)

    return {"rules": rules, "rpm": rpm, "target": data["target"]}


def extract(config, load_dir, save_dir):
    samples = {}

    # generate a list of all files in load_dir with suffix ".json"
    file_list = glob.glob(os.path.join(load_dir, "*test*.json"))

    print(file_list)
    for file in file_list:

        with open(file) as f:
            data = json.load(f)
            if data["panels"][0]["structure"] == config:

                # write the last number of the string file into i
                i = re.findall(r"\d+", file)[-1]

                samples[i] = get_sample(data)

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

#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import json

import yaml


def save_yaml(path, text):
    """parse string as yaml then dump as a file"""
    with open(path, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)


def load_json(path):
    with open(path) as f:
        d = json.load(f)
    return d


def save_json(path, d):
    with open(path, "w") as fp:
        json.dump(d, fp, sort_keys=True, indent=4)

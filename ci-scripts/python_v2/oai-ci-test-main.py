"""
Licensed to the OpenAirInterface (OAI) Software Alliance.

This code is licensed to OSA under one or more contributor license agreements.
See the NOTICE file distributed with this work for additional information
regarding copyright ownership.
The OpenAirInterface Software Alliance licenses this file to You under
the OAI Public License, Version 1.1  (the "License"); you may not use this file
except in compliance with the License.
You may obtain a copy of the License at

    http://www.openairinterface.org/?page_id=698

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------
For more information about the OpenAirInterface (OAI) Software Alliance:

    contact@openairinterface.org

"""

import argparse

import yaml

import cls_cn
import cls_ran
import cls_ue


def _parse_args() -> argparse.Namespace:
    """Parse the command line args

    Returns:
        argparse.Namespace: the created parser
    """
    parser = argparse.ArgumentParser(description='OAI CI Test Framework')

    # Infra YML filename
    parser.add_argument(
        '--infra_yaml', '-in',
        action='store',
        required=True,
        help='Setup Infrastructure Yaml File',
    )
    # Test Configuration YML filename
    parser.add_argument(
        '--tstcfg_yaml', '-tc',
        action='store',
        required=True,
        help='Test Configuration Yaml File',
    )
    # Git Information YML filename
    parser.add_argument(
        '--git_yaml', '-g',
        action='store',
        required=True,
        help='Git Information Yaml File',
    )
    # Mode
    parser.add_argument(
        '--mode',
        action='store',
        required=True,
        choices=['BuildAndTest', 'RetrieveLogs'],
        help='OAI CI Test Mode',
    )
    return parser.parse_args()


def get_test_infrastructure(filename):
    """
    Load the test infrastructure.

    Args:
        filename: yaml description file of test infrastructure

    Returns:
        infra: infrastructure data model
    """
    with open(filename, 'r') as infra_yml:
        infra = yaml.safe_load(infra_yml)
    return infra


def get_test_config(filename):
    """
    Load the test configuration data model.

    Args:
        filename: yaml description file of testcase

    Returns:
        test_config: test configuration data model
    """
    with open(filename, 'r') as test_yml:
        test_config = yaml.safe_load(test_yml)
    return test_config


def get_git_info(filename):
    """
    Load the git information data model.

    Args:
        filename: yaml description file of git information

    Returns:
        test_config: git information data model
    """
    with open(filename, 'r') as git_yml:
        git_info = yaml.safe_load(git_yml)
    return git_info


def get_test_objects(key, infra, test_cfg, git_info):
    """
    Load the test objects.

    Args:
        key: relevant keys to select
        infra: infrastructure data model
        test_cfg: test configuration data model
        git_info: git information data model

    Returns:
        dict_obj: dictionary of objects under test
    """
    # identify the relevant infra part from test config naming ex : RAN_0
    part = test_cfg['config'][key]['key']
    # retrieve the effective elements under test
    elements = test_cfg['config'][key][key].keys()

    # create dict of Objects under test
    dict_obj = {}
    for elt in elements:
        deploy = test_cfg['config'][key][key][elt]['Deploy']
        # retrieve the infra part of the element under test only
        obj_part = infra[part][elt]
        if key == 'RAN':
            dict_obj[elt] = cls_ran.NodeB(infra, obj_part, deploy, git_info)
        elif key == 'CN':
            dict_obj[elt] = cls_cn.CN(infra, obj_part, deploy, git_info)
        elif key == 'UE':
            dict_obj[elt] = cls_ue.UE(infra, obj_part, deploy, git_info)
        else:
            pass
    return dict_obj


if __name__ == '__main__':
    # Parse the arguments to recover the YAML filenames
    args = _parse_args()
    # Retrieve the infrastructure
    infrastructure = get_test_infrastructure(args.infra_yaml)
    # Retrieve the test configuration (ie infra being used and testsuite)
    test_cfg = get_test_config(args.tstcfg_yaml)
    # Retrieve the git information
    git_info = get_git_info(args.git_yaml)
    # Populate objects
    RAN = get_test_objects('RAN', infrastructure, test_cfg, git_info)
    CN = get_test_objects('CN', infrastructure, test_cfg, git_info)
    UEs = get_test_objects('UE', infrastructure, test_cfg, git_info)
    for key1 in RAN.keys():
        print(key1, RAN[key1].Type)
    if args.mode == 'BuildAndTest':
        print('Mode is BuildAndTest')
    if args.mode == 'RetrieveLogs':
        print('Mode is RetrieveLogs')

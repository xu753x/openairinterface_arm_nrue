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

import yaml

import cls_cn
import cls_ran
import cls_ue


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


def get_test_objects(key, infrastructure, test_cfg):
    """
    Load the test objects.

    Args:
        key: relevant keys to select
        infrastructure: infrastructure data model
        test_cfg: test configuration data model

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
        deployment = test_cfg['config'][key][key][elt]['Deploy']
        # retrieve the infra part of the element under test only
        obj_part = infrastructure[part][elt]
        if key == 'RAN':
            dict_obj[elt] = cls_ran.NodeB(infrastructure, obj_part, deployment)
        elif key == 'CN':
            dict_obj[elt] = cls_cn.CN(infrastructure, obj_part, deployment)
        elif key == 'UE':
            dict_obj[elt] = cls_ue.UE(infrastructure, obj_part, deployment)
        else:
            pass
    return dict_obj


if __name__ == '__main__':
    testbench = 'testinfra-as-code.yaml'
    test = 'test-example.yaml'
    infrastructure = get_test_infrastructure(testbench)
    test_cfg = get_test_config(test)
    RAN = get_test_objects('RAN', infrastructure, test_cfg)
    CN = get_test_objects('CN', infrastructure, test_cfg)
    UEs = get_test_objects('UE', infrastructure, test_cfg)

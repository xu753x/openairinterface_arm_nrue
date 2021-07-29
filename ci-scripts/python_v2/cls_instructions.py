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


class Instructions:
    def __init__(self):
        self.test_dict = {'Build_PhySim' : self.function1 , 'Run_PhySim' : self.function2}
        
    def function1(self,RAN,CN,UEs):
        print("executing from method function1")
        for k in RAN:
            print(RAN[k].Type, RAN[k].Name)
            if RAN[k].Type=='eNB':
                RAN[k].PrintDeploy()
            elif RAN[k].Type=='gNB':
                RAN[k].PrintGitInfo()
            else:
                print("invalid type")
        
    def function2(self,RAN,CN,UEs):
        print("executing from method function2")
        
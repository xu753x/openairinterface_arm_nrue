#loading xml action list from yaml
import yaml

import cls_ue
import cls_ran
import cls_cn


#from yaml description of test infrastructure, return infrastructure data model
def GetTestInfrastructure(filename):

    with open(filename,'r') as f:
        infra = yaml.load(f)
    return infra
    
    
#from yaml description of testcase, return configuration data model
def GetTestConfig(filename):

    with open(filename,'r') as f:
        test_config = yaml.load(f)   
    return test_config 



def GetTestObjects(key,infrastructure,test_cfg):
   
    part=test_cfg['config'][key]['key'] #identify the relevant infra part from test config naming ex : RAN_0
    elements=test_cfg['config'][key][key].keys() #retrieve the effective elements under test

    #create dict of Objects under test
    OBJ={}
    for n in elements:
        deployment = test_cfg['config'][key][key][n]['Deploy']          
        obj_part=infrastructure[part][n] #retrieve the infra part of the element under test only
        if key=='RAN':
            OBJ[n]=cls_ran.NodeB(infrastructure,obj_part,deployment)  
        elif key=='CN':
            OBJ[n]=cls_cn.CN(infrastructure,obj_part,deployment)
        elif key=='UE':    
            OBJ[n]=cls_ue.UE(infrastructure,obj_part,deployment)
        else:
            pass
    return OBJ #dictionary of objects under test
    


if __name__ == "__main__":
    testbench='testinfra-as-code.yaml'
    test='test-example.yaml'
    infrastructure=GetTestInfrastructure(testbench)
    test_cfg=GetTestConfig(test)
    RAN=GetTestObjects('RAN',infrastructure,test_cfg)
    CN=GetTestObjects('CN',infrastructure,test_cfg)
    UEs=GetTestObjects('UE',infrastructure,test_cfg)


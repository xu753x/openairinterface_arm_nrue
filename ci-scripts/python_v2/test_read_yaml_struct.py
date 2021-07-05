#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 09:38:02 2021

@author: hardy
"""


class NodeB:
    def __init__(self,infra,config,deployment):
        self.deployment=deployment
        self.infra=infra
        for k,v in config.items():
            setattr(self,k,v)
    
class UE:
    def __init__(self,infra,config):
        self.infra=infra
        for k,v in config.items():
            setattr(self,k,v)              

#loading xml action list from yaml
import yaml
yaml_file='testinfra-as-code.yaml'
with open(yaml_file,'r') as f:
    infra = yaml.load(f)
    
    
yaml_file='test-example.yaml'
with open(yaml_file,'r') as f:
    test = yaml.load(f)    
    

#RAN spec from test    
ran_key=test['config']['RAN']['key']
nodes_key=test['config']['RAN']['Nodes'].keys()

 #create dict of RAN objects RAN under Test
RAN={}
for n in nodes_key:
    deployment = test['config']['RAN']['Nodes'][n]['Deploy']
    config=infra[ran_key][n]
    RAN[n]=NodeB(infra,config,deployment)  
    
    

#UE spec from test
ue_key=test['config']['UE']['key']
allues_key=test['config']['UE']['UEs'].keys()

#create dict of UE objects UE under Test
UEs={}
for u in allues_key:
    config=infra[ue_key][u]
    UEs[u]=UE(infra,config)  
    





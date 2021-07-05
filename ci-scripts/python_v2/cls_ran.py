class NodeB:
    def __init__(self,infra,config,deployment):
        self.Deployment=deployment
        self.Infra=infra
        for k,v in config.items():
            setattr(self,k,v)
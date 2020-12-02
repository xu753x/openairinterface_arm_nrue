#docker build -t ci-py:test -f py-image.Dockerfile --build-arg NEEDED_GIT_PROXY="http://proxy.eurecom.fr:8080" .
#docker run -it --name prod-ci-py ci-py:test /bin/bash

FROM ubuntu:18.04
ARG NEEDED_GIT_PROXY

RUN apt-get update \
    && apt install git -y

# In some network environments, GIT proxy is required
RUN /bin/bash -c "if [[ -v NEEDED_GIT_PROXY ]]; then git config --global http.proxy $NEEDED_GIT_PROXY; fi"

RUN apt install python3 -y \
    && apt install python3-pip -y \
    #packages required by framework
    && pip3 install pexpect \
    && pip3 install pyyaml \
    #functions required by framework call to subprocess
    #curl, fromdos, ping, jq

    && apt install curl -y \
    && apt install tofrodos -y \
    && apt install iputils-ping -y \
    && apt install jq -y \

    #clone repo to retrieve python framework files
    && git clone -b rh_fr1_newjenkins https://gitlab.eurecom.fr/oai/openairinterface5g.git /tmp/CI-PY 
 
ENTRYPOINT /bin/bash

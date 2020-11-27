<table style="border-collapse: collapse; border: none;">
  <tr style="border-collapse: collapse; border: none;">
    <td style="border-collapse: collapse; border: none;">
      <a href="http://www.openairinterface.org/">
         <img src="../doc/images/oai_final_logo.png" alt="" border=3 height=50 width=150>
         </img>
      </a>
    </td>
    <td style="border-collapse: collapse; border: none; vertical-align: center;">
      <b><font size = "5">OAI Docker/Podman Build and Usage Procedures</font></b>
    </td>
  </tr>
</table>

---

**Table of Contents**

1. [Build Strategy](#1-build-strategy)
2. [File organization](#2-file-organization)
3. [Building using docker under Ubuntu 18.04](#3-building-using-docker-under-ubuntu-1804)
4. [Building using podman under Red Hat Entreprise Linux 8.2](#4-building-using-podman-under-red-hat-entreprise-linux-82)
5. [Running modems using docker under Ubuntu 18.04](#5-running-modems-using-docker-under-ubuntu-1804)
6. [Running modems using podman under Red Hat Entreprise Linux 8.2](#6-running-modems-using-podman-under-red-hat-entreprise-linux-82)

---

# 1. Build Strategy #

For all platforms, the strategy for building docker/podman images is the same:

*  First we create a common shared image that contains:
   -  the latest source files (by using the `COPY` function)
   -  all the means to build an OAI RAN executable
      *  all packages, compilers, ...
      *  especially UHD is installed 
*  Then from this shared image (`ran-build`) we can build target images for:
   -  eNB
   -  gNB
   -  lte-UE
   -  nr-UE
*  These target images will only contain:
   -  the generated executable (for example `lte-softmodem.Rel15`)
   -  the generated shared libraries (for example `liboai_usrpdevif.so.Rel15`)
   -  the needed libraries and packages to run these generated binaries
   -  Some configuration file templates
   -  Some tools (such as `ping`, `ifconfig`)

TO DO:

-  Proper entrypoints
-  Proper port exposure
-  ...

# 2. File organization #

Dockerfiles are named with the following naming convention: `Dockerfile.${target}.${OS-version}.${cluster-version}`

Targets can be:

-  `ran` for an image named `ran-build` (the shared image)
-  `eNB` for an image named `oai-enb`
-  `gNB` for an image named `oai-gnb`
-  `lteUE` for an image named `oai-lte-ue`
-  `nrUE` for an image named `oai-nr-ue`

The currently-supported OS are:

- `rhel8.2` for Red Hat Entreprise Linux
- `ubuntu18` for Ubuntu 18.04 LTS

The currently-supported cluster version is:

- `rhel8.2.oc4-4`

We have also `rhel7.oc4-4` support but it will be discontinued soon.

For more details in build within a Openshift Cluster, see [OpenShift README](../openshift/README.md) for more details.

# 3. Building using `docker` under Ubuntu 18.04 #

## 3.1. Pre-requisites ##

* `git` installed
* `docker-ce` installed
* Pulling `ubuntu:bionic` from DockerHub

## 3.2. Building the shared image ##

This can be done starting `2020.w41` tag on the `develop` branch, or any branch that includes that tag.

```bash
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
cd openairinterface5g
git checkout develop
```

In our Eurecom/OSA environment we need to pass a GIT proxy.

```bash
docker build --target ran-build --tag ran-build:latest --file docker/Dockerfile.ran.ubuntu18 --build-arg NEEDED_GIT_PROXY="http://proxy.eurecom.fr:8080" .
```

if you don't need it, do NOT pass any value:

```bash
docker build --target ran-build --tag ran-build:latest --file docker/Dockerfile.ran.ubuntu18 .
```

After a while:

```bash
docker image ls
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
ran-build           latest              ccb721bc0b57        1 minute ago        4.06GB
...
```

## 3.3. Building any target image ##

For example, the eNB:

```bash
docker build --target oai-enb --tag oai-enb:latest --file docker/Dockerfile.eNB.ubuntu18 .
```

After a while:

```
docker image ls
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
oai-enb             latest              25ddbd8b7187        1 minute ago        516MB
<none>              <none>              875ea3b05b60        8 minutes ago       8.18GB
ran-build           latest              ccb721bc0b57        1 hour ago          4.06GB
```

Do not forget to remove the temporary image:

```
docker image prune --force
```

# 4. Building using `podman` under Red Hat Entreprise Linux 8.2 #

TODO.

# 5. Running modems using `docker` under Ubuntu 18.04 #

TODO.

# 6. Running modems using `podman` under Red Hat Entreprise Linux 8.2 #

TODO.

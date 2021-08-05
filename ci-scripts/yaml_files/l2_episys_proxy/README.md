<table style="border-collapse: collapse; border: none;">
  <tr style="border-collapse: collapse; border: none;">
    <td style="border-collapse: collapse; border: none;">
      <a href="http://www.openairinterface.org/">
         <img src="../../../doc/images/oai_final_logo.png" alt="" border=3 height=50 width=150>
         </img>
      </a>
    </td>
    <td style="border-collapse: collapse; border: none; vertical-align: center;">
      <b><font size = "5">OAI Full Stack L2-nFAPI simulation with containers</font></b>
    </td>
  </tr>
</table>

This page is only valid for an `Ubuntu18` host.

**CAUTION: this is very experimental on the RAN side. For the moment, on the `episys-merge` branch.**

**TABLE OF CONTENTS**

1. [Retrieving the images on Docker-Hub](#1-retrieving-the-images-on-docker-hub)
   1. [Building the RAN docker images](#11-building-the-ran-docker-images)
   2. [Building the proxy docker image](#12-building-the-proxy-docker-image)
2. [Deploy containers](#2-deploy-containers)
   1. [Deploy and Configure Cassandra Database](#21-deploy-and-configure-cassandra-database)
   2. [Deploy OAI CN4G containers](#22-deploy-oai-cn4g-containers)
   3. [Deploy OAI eNB in VNF L2 nFAPI emulator mode](#23-deploy-oai-enb-in-vnf-l2-nfapi-emulator-mode)
   4. [Deploy the VNF Proxy container](#24-deploy-the-vnf-proxy-container)
   5. [Deploy OAI LTE UE in VNF L2-NFAPI emulator mode](#25-deploy-oai-lte-ue-in-vnf-l2-nfapi-emulator-mode)
3. [Check traffic](#3-check-traffic)
4. [Un-deployment](#4-un-deployment)

# 1. Retrieving the images on Docker-Hub #

Currently the images are hosted under the user account `rdefosseoai`.

This may change in the future.

Once again you may need to log on [docker-hub](https://hub.docker.com/) if your organization has reached pulling limit as `anonymous`.

```bash
$ docker login
Login with your Docker ID to push and pull images from Docker Hub. If you don't have a Docker ID, head over to https://hub.docker.com to create one.
Username:
Password:
```

Now pull images.

```bash
$ docker pull cassandra:2.1
$ docker pull rdefosseoai/oai-hss:latest
$ docker pull rdefosseoai/oai-mme:latest
$ docker pull rdefosseoai/oai-spgwc:latest
$ docker pull rdefosseoai/oai-spgwu-tiny:latest
```

And **re-tag** them for tutorials' docker-compose file to work.

```bash
$ docker image tag rdefosseoai/oai-spgwc:latest oai-spgwc:latest
$ docker image tag rdefosseoai/oai-hss:latest oai-hss:latest
$ docker image tag rdefosseoai/oai-spgwu-tiny:latest oai-spgwu-tiny:latest
$ docker image tag rdefosseoai/oai-mme:latest oai-mme:latest
```

How to build the Traffic-Generator image is explained [here](https://github.com/OPENAIRINTERFACE/openair-epc-fed/blob/master/docs/GENERATE_TRAFFIC.md#1-build-a-traffic-generator-image).

## 1.1. Building the RAN docker images ##

As it is experimental on the `episys-merge` branch, RAN images on Docker-Hub won't do. Once this branch is merged into `develop`, I will push images again and update tutorial.

So for the moment, you SHALL build your-self images. See [how to build the RAN docker images](../../../docker/README.md).

Do not forget to re-tag the generated images with `develop` tag.

```bash
$ docker image tag oai-enb:latest oai-enb:develop
$ docker image tag oai-lte-ue:latest oai-lte-ue:develop
```

## 1.2. Building the proxy docker image ##

The proxy code is hosted on GitHub.com by the `EpiSys Science, Inc.` team.

```bash
$ git clone https://github.com/EpiSci/oai-lte-multi-ue-proxy.git
$ cd oai-lte-multi-ue-proxy
$ git rebase origin/master
$ docker build --no-cache --target oai-lte-multi-ue-proxy --tag oai-lte-multi-ue-proxy:latest --file docker/Dockerfile.ubuntu18.04 .
$ docker image prune --force
```

## 1.3. Logout if needed ##

```bash
$ docker logout
```

# 2. Deploy containers #

**CAUTION: this SHALL be done in multiple steps.**

**Just `docker-compose up -d` WILL NOT WORK!**

**All commands SHALL be executed in the `ci-scripts/yaml_files/l2_episys_proxy` folder!**

## 2.1. Deploy and Configure Cassandra Database ##

It is very crutial that the Cassandra DB is fully configured before you do anything else!

```bash
$ docker-compose -f docker-compose-01-users-epc.yml up -d db_init
Creating network "l2sim-net" with the default driver
Creating l2sim-cassandra ... done
Creating l2sim-db-init   ... done


$ docker logs l2sim-db-init --follow
Connection error: ('Unable to connect to any servers', {'192.168.61.2': error(111, "Tried connecting to [('192.168.61.2', 9042)]. Last error: Connection refused")})
...
Connection error: ('Unable to connect to any servers', {'192.168.61.2': error(111, "Tried connecting to [('192.168.61.2', 9042)]. Last error: Connection refused")})
OK
```

**You SHALL wait until you HAVE the `OK` message in the logs!**

```bash
$ docker rm l2sim-db-init
```

At this point, you can prepare a capture on the newly-created public docker bridge:

```bash
$ ifconfig l2sim-net
        inet 192.168.61.1  netmask 255.255.255.192  broadcast 192.168.61.63
        inet6 fe80::42:aff:fe1e:e5dd  prefixlen 64  scopeid 0x20<link>
        ether 02:42:0a:1e:e5:dd  txqueuelen 0  (Ethernet)
        RX packets 7  bytes 196 (196.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 5  bytes 446 (446.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
$ sudo tshark -i l2sim-net -f 'port 3868 or port 2123 or port 36412 or port 36422 or port 46520 or port 8805' -w /tmp/my-oai-control-plane.pcap
```

**BE CAREFUL: please use that filter or you will also capture the data-plane with IQ samples between `eNB` and `LTE-UE`.**

**and your capture WILL become huge (10s of Gbytes).**

## 2.2. Deploy OAI CN4G containers ##

```bash
$ docker-compose -f docker-compose-01-users-epc.yml up -d oai_mme oai_spgwu trf_gen
l2sim-cassandra is up-to-date
Creating l2sim-trf-gen   ... done
Creating l2sim-oai-hss ... done
Creating l2sim-oai-mme ... done
Creating l2sim-oai-spgwc ... done
Creating l2sim-oai-spgwu-tiny ... done
```

You shall wait until all containers are `healthy`. About 10 seconds!

```bash
$ docker-compose -f docker-compose-01-users-epc.yml ps -a
        Name                      Command                  State                            Ports                      
-----------------------------------------------------------------------------------------------------------------------
l2sim-cassandra        docker-entrypoint.sh cassa ...   Up (healthy)   7000/tcp, 7001/tcp, 7199/tcp, 9042/tcp, 9160/tcp
l2sim-oai-hss          /openair-hss/bin/entrypoin ...   Up (healthy)   5868/tcp, 9042/tcp, 9080/tcp, 9081/tcp          
l2sim-oai-mme          /openair-mme/bin/entrypoin ...   Up (healthy)   2123/udp, 3870/tcp, 5870/tcp                    
l2sim-oai-spgwc        /openair-spgwc/bin/entrypo ...   Up (healthy)   2123/udp, 8805/udp                              
l2sim-oai-spgwu-tiny   /openair-spgwu-tiny/bin/en ...   Up (healthy)   2152/udp, 8805/udp                              
l2sim-trf-gen          /bin/bash -c ip route add  ...   Up (healthy)                                                   
```

## 2.3. Deploy OAI eNB in VNF L2 nFAPI emulator mode ##

```bash
$ docker-compose -f docker-compose-01-users-epc.yml up -d enb_vnf_nfapi
Creating l2sim-enb-vnf ... done
```

Again wait for the healthy state:

```bash
$ docker-compose -f docker-compose-01-users-epc.yml ps -a
        Name                      Command                  State                            Ports                      
-----------------------------------------------------------------------------------------------------------------------
l2sim-cassandra        docker-entrypoint.sh cassa ...   Up (healthy)   7000/tcp, 7001/tcp, 7199/tcp, 9042/tcp, 9160/tcp
l2sim-enb-vnf          /opt/oai-enb/bin/entrypoin ...   Up (healthy)   2152/udp, 36412/udp, 36422/udp                  
l2sim-oai-hss          /openair-hss/bin/entrypoin ...   Up (healthy)   5868/tcp, 9042/tcp, 9080/tcp, 9081/tcp          
l2sim-oai-mme          /openair-mme/bin/entrypoin ...   Up (healthy)   2123/udp, 3870/tcp, 5870/tcp                    
l2sim-oai-spgwc        /openair-spgwc/bin/entrypo ...   Up (healthy)   2123/udp, 8805/udp                              
l2sim-oai-spgwu-tiny   /openair-spgwu-tiny/bin/en ...   Up (healthy)   2152/udp, 8805/udp                              
l2sim-trf-gen          /bin/bash -c ip route add  ...   Up (healthy)                           
```

Check if the eNB connected to MME:

```bash
$ docker logs rfsim4g-oai-mme
...
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0039    ======================================= STATISTICS ============================================

DEBUG MME-AP src/mme_app/mme_app_statistics.c:0042                   |   Current Status| Added since last display|  Removed since last display |
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0048    Connected eNBs |          0      |              0              |             0               |
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0054    Attached UEs   |          0      |              0              |             0               |
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0060    Connected UEs  |          0      |              0              |             0               |
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0066    Default Bearers|          0      |              0              |             0               |
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0072    S1-U Bearers   |          0      |              0              |             0               |

DEBUG MME-AP src/mme_app/mme_app_statistics.c:0075    ======================================= STATISTICS ============================================

DEBUG SCTP   rc/sctp/sctp_primitives_server.c:0469    Client association changed: 0
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0101    ----------------------
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0102    SCTP Status:
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0103    assoc id .....: 675
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0104    state ........: 4
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0105    instrms ......: 2
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0106    outstrms .....: 2
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0108    fragmentation : 1452
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0109    pending data .: 0
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0110    unack data ...: 0
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0111    rwnd .........: 106496
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0112    peer info     :
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0114        state ....: 2
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0116        cwnd .....: 4380
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0118        srtt .....: 0
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0120        rto ......: 3000
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0122        mtu ......: 1500
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0123    ----------------------
DEBUG SCTP   rc/sctp/sctp_primitives_server.c:0479    New connection
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0205    ----------------------
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0206    Local addresses:
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0217        - [192.168.61.4]
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0234    ----------------------
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0151    ----------------------
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0152    Peer addresses:
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0163        - [192.168.61.20]
DEBUG SCTP   enair-mme/src/sctp/sctp_common.c:0178    ----------------------
DEBUG SCTP   rc/sctp/sctp_primitives_server.c:0554    SCTP RETURNING!!
DEBUG SCTP   rc/sctp/sctp_primitives_server.c:0547    [675][44] Msg of length 51 received from port 36412, on stream 0, PPID 18
DEBUG SCTP   rc/sctp/sctp_primitives_server.c:0554    SCTP RETURNING!!
DEBUG S1AP   mme/src/s1ap/s1ap_mme_handlers.c:2826    Create eNB context for assoc_id: 675
DEBUG S1AP   mme/src/s1ap/s1ap_mme_handlers.c:0361    S1-Setup-Request macroENB_ID.size 3 (should be 20)
DEBUG S1AP   mme/src/s1ap/s1ap_mme_handlers.c:0321    New s1 setup request incoming from macro eNB id: 00e01
DEBUG S1AP   mme/src/s1ap/s1ap_mme_handlers.c:0423    Adding eNB to the list of served eNBs
DEBUG S1AP   mme/src/s1ap/s1ap_mme_handlers.c:0438    Adding eNB id 3585 to the list of served eNBs
DEBUG SCTP   rc/sctp/sctp_primitives_server.c:0283    [44][675] Sending buffer 0x7f9394009f90 of 27 bytes on stream 0 with ppid 18
DEBUG SCTP   rc/sctp/sctp_primitives_server.c:0296    Successfully sent 27 bytes on stream 0
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0039    ======================================= STATISTICS ============================================

DEBUG MME-AP src/mme_app/mme_app_statistics.c:0042                   |   Current Status| Added since last display|  Removed since last display |
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0048    Connected eNBs |          1      |              1              |             0               |
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0054    Attached UEs   |          0      |              0              |             0               |
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0060    Connected UEs  |          0      |              0              |             0               |
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0066    Default Bearers|          0      |              0              |             0               |
DEBUG MME-AP src/mme_app/mme_app_statistics.c:0072    S1-U Bearers   |          0      |              0              |             0               |

DEBUG MME-AP src/mme_app/mme_app_statistics.c:0075    ======================================= STATISTICS ============================================
...
```

## 2.4. Deploy the VNF Proxy container ##

```bash
$ docker-compose -f docker-compose-01-users-epc.yml up -d proxy_pnf_nfapi
Creating l2sim-proxy-pnf ... done
```

Again wait for the healthy state:

```bash
$ docker-compose -f docker-compose-01-users-epc.yml ps -a
        Name                      Command                  State                            Ports                      
-----------------------------------------------------------------------------------------------------------------------
l2sim-cassandra        docker-entrypoint.sh cassa ...   Up (healthy)   7000/tcp, 7001/tcp, 7199/tcp, 9042/tcp, 9160/tcp
l2sim-enb-vnf          /opt/oai-enb/bin/entrypoin ...   Up (healthy)   2152/udp, 36412/udp, 36422/udp                  
l2sim-oai-hss          /openair-hss/bin/entrypoin ...   Up (healthy)   5868/tcp, 9042/tcp, 9080/tcp, 9081/tcp          
l2sim-oai-mme          /openair-mme/bin/entrypoin ...   Up (healthy)   2123/udp, 3870/tcp, 5870/tcp                    
l2sim-oai-spgwc        /openair-spgwc/bin/entrypo ...   Up (healthy)   2123/udp, 8805/udp                              
l2sim-oai-spgwu-tiny   /openair-spgwu-tiny/bin/en ...   Up (healthy)   2152/udp, 8805/udp                              
l2sim-proxy-pnf        /bin/bash -c /oai-lte-mult ...   Up (healthy)                                                   
l2sim-trf-gen          /bin/bash -c ip route add  ...   Up (healthy)                           
```

## 2.5. Deploy OAI LTE UE in VNF L2-NFAPI emulator mode ##

```bash
$ docker-compose -f docker-compose-01-users-epc.yml up -d oai_ue_l2_sim0
Creating l2sim-oai-ue0 ... done
```

Last wait for the healthy state:

```bash
$ docker-compose -f docker-compose-01-users-epc.yml ps -a
        Name                      Command                  State                            Ports                      
-----------------------------------------------------------------------------------------------------------------------
l2sim-cassandra        docker-entrypoint.sh cassa ...   Up (healthy)   7000/tcp, 7001/tcp, 7199/tcp, 9042/tcp, 9160/tcp
l2sim-enb-vnf          /opt/oai-enb/bin/entrypoin ...   Up (healthy)   2152/udp, 36412/udp, 36422/udp                  
l2sim-oai-hss          /openair-hss/bin/entrypoin ...   Up (healthy)   5868/tcp, 9042/tcp, 9080/tcp, 9081/tcp          
l2sim-oai-mme          /openair-mme/bin/entrypoin ...   Up (healthy)   2123/udp, 3870/tcp, 5870/tcp                    
l2sim-oai-spgwc        /openair-spgwc/bin/entrypo ...   Up (healthy)   2123/udp, 8805/udp                              
l2sim-oai-spgwu-tiny   /openair-spgwu-tiny/bin/en ...   Up (healthy)   2152/udp, 8805/udp                              
l2sim-oai-ue0          /opt/oai-lte-ue/bin/entryp ...   Up (healthy)                                                   
l2sim-proxy-pnf        /bin/bash -c /oai-lte-mult ...   Up (healthy)                                                   
l2sim-trf-gen          /bin/bash -c ip route add  ...   Up (healthy)                                                   
```

Making sure the OAI UE is connected:

On the LTE UE:

```bash
$ docker exec l2sim-oai-ue0 /bin/bash -c "ifconfig"
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.61.30  netmask 255.255.255.192  broadcast 192.168.61.63
        ether 02:42:c0:a8:3d:1e  txqueuelen 0  (Ethernet)
        RX packets 81385  bytes 4809459 (4.8 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 61628  bytes 3213169 (3.2 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 0  bytes 0 (0.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 0  bytes 0 (0.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

oaitun_ue1: flags=4305<UP,POINTOPOINT,RUNNING,NOARP,MULTICAST>  mtu 1500
        inet 12.1.1.2  netmask 255.0.0.0  destination 12.1.1.2
        unspec 00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00  txqueuelen 500  (UNSPEC)
        RX packets 0  bytes 0 (0.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 0  bytes 0 (0.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

The tunnel `oaitun_ue1` SHALL be mounted and with an IP address in the `12.1.1.xxx` range.

# 3. Check traffic #

```bash
$ docker exec l2sim-oai-ue0 /bin/bash -c "ping -c 2 www.lemonde.fr"
PING s2.shared.global.fastly.net (151.101.122.217) 56(84) bytes of data.
64 bytes from 151.101.122.217 (151.101.122.217): icmp_seq=1 ttl=54 time=12.9 ms
64 bytes from 151.101.122.217 (151.101.122.217): icmp_seq=2 ttl=54 time=12.9 ms

--- s2.shared.global.fastly.net ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1001ms
rtt min/avg/max/mdev = 12.948/12.966/12.985/0.115 ms
$ docker exec l2sim-oai-ue0 /bin/bash -c "ping -I oaitun_ue1 -c 2 www.lemonde.fr"
PING s2.shared.global.fastly.net (151.101.122.217) from 12.1.1.2 oaitun_ue1: 56(84) bytes of data.
64 bytes from 151.101.122.217 (151.101.122.217): icmp_seq=1 ttl=53 time=42.2 ms
64 bytes from 151.101.122.217 (151.101.122.217): icmp_seq=2 ttl=53 time=28.2 ms

--- s2.shared.global.fastly.net ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1001ms
rtt min/avg/max/mdev = 28.279/35.262/42.245/6.983 ms
```

The 1st ping command is NOT using the OAI stack. My network infrastructure has a response of `13 ms` to reach this website.

The 2nd ping command is using the OAI stack. So the stack takes `35.2 - 12.9 = 22.9 ms`.

# 4. Un-deployment #

```bash
$ docker-compose -f docker-compose-01-users-epc.yml down
Stopping l2sim-oai-ue0        ... done
Stopping l2sim-proxy-pnf      ... done
Stopping l2sim-enb-vnf        ... done
Stopping l2sim-oai-spgwu-tiny ... done
Stopping l2sim-oai-spgwc      ... done
Stopping l2sim-oai-mme        ... done
Stopping l2sim-oai-hss        ... done
Stopping l2sim-trf-gen        ... done
Stopping l2sim-cassandra      ... done
Removing l2sim-oai-ue0        ... done
Removing l2sim-proxy-pnf      ... done
Removing l2sim-enb-vnf        ... done
Removing l2sim-oai-spgwu-tiny ... done
Removing l2sim-oai-spgwc      ... done
Removing l2sim-oai-mme        ... done
Removing l2sim-oai-hss        ... done
Removing l2sim-trf-gen        ... done
Removing l2sim-cassandra      ... done
Removing network l2sim-net
```

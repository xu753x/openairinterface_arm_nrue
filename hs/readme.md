> 目前可运行的版本是 perf cufft(symbol)

# 生成动态链接库

```bash
# .c .cpp .cu -> .o 
nvcc --compiler-options "-fPIC" -c ./cuFFT3.cu -o ./cuFFT.o
# .o -> .so
gcc -shared -o cuFFT.so *.o -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcufft
# 复制到合适的位置
cp cuFFT.so /home/witcomm/hs/code/ran/targets/PROJECTS/GENERIC-NR-5GC/CONF/
```



# 更改OAI配置文件

对使用的配置文件进行修改`worker_config   = "WORKER_DISABLE";`，使cpu单线程运行。比如现在使用的配置文件是`gnb.sa.band78.fr1.106PRB.usrpb210.conf`

![image-20210608125616795](https://cdn.jsdelivr.net/gh/cucengineer/draw.io/img/20210608125617.png)



  # dft与改进2：cuFFT3.so时间对比

  ![image-20210608124450894](https://cdn.jsdelivr.net/gh/cucengineer/draw.io/img/20210608131904.png)

  

图1统计的是一个symbol内FFT的时间，图1上为dft，时间为5.5us；图1下为cudft2048，时间为33us。

![symbol](https://cdn.jsdelivr.net/gh/cucengineer/draw.io/img/20210608131915.png)

  <center> 图1 symbol
------


![image-20210608124547254](https://cdn.jsdelivr.net/gh/cucengineer/draw.io/img/20210608131915.png)

 

 图2统计的是一个slot内包含FFT处理的时间，图2上为dft，时间为87us；图2下为cudft2048，时间为482us。 

![slot](https://cdn.jsdelivr.net/gh/cucengineer/draw.io/img/20210608131921.png)

  <center> 图2 slot
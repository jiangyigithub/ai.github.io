---
title: Nvidia-GPU specs
description: 本文汇总了NVIDIA GPU 系列的技术规格以及关键改进
date: 2026-01-14 11:09:19+0800
math: true
tags: 
    - NVIDIA
    - GPU
categories:
    - Infra
---


## V100

### V100 关键改进

- Volta architecture
- SM architecture: 支持深度学习
- 2nd NVIDIA NVLink
- HBM2 memory
- Volta Multi-process Service

### V100 技术规格

| Tesla Product             | Tesla K40  | Tesla M40   | Tesla P100 | Tesla V100 |
| ------------------------- | -------------- | --------------- | -------------- | -------------- |
| GPU                       | GK180 (Kepler) | GM200 (Maxwell) | GP100 (Pascal) | GV100 (Volta)  |
| SMs                   | 15             | 24              | 56             | 80             |
| TPCs                  | 15             | 24              | 28             | 40             |
| FP32 Cores / GPU      | 2880           | 3072            | 3584           | 5120           |
| FP64 Cores / GPU      | 960            | 96              | 1792           | 2560           |
| Tensor Cores / GPU    | NA             | NA              | NA             | 640            |
| GPU Boost Clock       | 810/875 MHz    | 1114 MHz        | 1480 MHz       | 1530 MHz       |
| Peak FP32 TFLOPS²     | 5              | 6.8             | 10.6           | 15.7           |
| Peak FP64 TFLOPS²     | 1.7            | .21             | 5.3            | 7.8            |
| Peak Tensor TFLOPS²   | NA             | NA              | NA             | 125            |
| Memory Size           | Up to 12 GB    | Up to 24 GB     | 16 GB          | 16 GB          |
| Memory Interface      | 384-bit GDDR5  | 384-bit GDDR5   | 4096-bit HBM2  | 4096-bit HBM2  |
| TDP                   | 235 Watts      | 250 Watts       | 300 Watts      | 300 Watts      |
| Manufacturing Process | 28 nm          | 28 nm           | 16 nm FinFET+  | 12 nm FFN      |

内存规格

| GPU                                 | Kepler GK180       | Maxwell GM200 | Pascal GP100 | Volta GV100              |
| ----------------------------------- | ------------------ | ------------- | ------------ | ------------------------ |
| Compute Capability                  | 3.5                | 5.2           | 6.0          | 7.0                      |
| Threads / Warp                      | 32                 | 32            | 32           | 32                       |
| Max Warps / SM                      | 64                 | 64            | 64           | 64                       |
| Max Threads / SM                    | 2048               | 2048          | 2048         | 2048                     |
| Max Thread Blocks / SM              | 32                 | 32            | 32           | 32                       |
| Max 32-bit Registers / SM           | 65536              | 65536         | 65536        | 65536                    |
| Max Registers / Block               | 65536              | 65536         | 65536        | 65536                    |
| Max Registers / Thread              | 255                | 255           | 255          | 255                      |
| Max Thread Block Size               | 1024               | 1024          | 1024         | 1024                     |
| FP32 Cores / SM                     | 192                | 128           | 64           | 64                       |
| Ratio of SM Registers to FP32 Cores | 341                | 512           | 1024         | 1024                     |
| Shared Memory Size / SM             | 16 KB/32 KB/ 48 KB | 96 KB         | 64 KB        | Configurable up to 96 KB |

系统规格

| Specification        | DGX-1 (Tesla P100)                | DGX-1 (Tesla V100)                             |
| ------------------------ | ------------------------------------- | -------------------------------------------------- |
| GPU                  | 8x Tesla P100 GPUs                    | 8x Tesla V100 GPUs                                 |
| TFLOPS               | 170 (GPU FP16) + 3 (CPU FP32)         | 1 (GPU Tensor PFLOP)                               |
| GPU Memory           | 16 GB per GPU / 128 GB per DGX-1 Node | 16 GB or 32 GB per GPU / 128-256 GB per DGX-1 Node |
| CPU                  | Dual 20-core Intel® Xeon® E5-2698 v4  | Dual 20-core Intel® Xeon® E5-2698 v4               |
| FP32 CUDA Cores      | 28,672 Cores                          | 40,960 Cores                                       |
| System Memory        | Up to 512 GB 2133 MHz DDR4 LRDIMM     | Up to 512 GB 2133 MHz DDR4 LRDIMM                  |
| Storage              | 4x 1.92 TB SSD RAID 0                 | 4x 1.92 TB SSD RAID 0                              |
| Network Interconnect | Dual 10 GbE, 4 IB EDR                 | Dual 10 GbE, 4 IB EDR                              |
| System Dimensions    | 866 D x 444 W x 131 H (mm)            | 866 D x 444 W x 131 H (mm)                         |
| System Weight        | 80 lbs                                | 80 lbs                                             |
| Max Power TDP        | 3200 W                                | 3200 W                                             |
| Operating Temp       | 10 - 35°C                             | 10 - 35°C                                          |

## A100

### A100 关键改进

- Ampere 架构：使用 MIG 来将 A100 切分为更小的实例或者链接更多 GPU
- Tensor Cores: 312 TFLOPs/s
- NVLink: 更高的 throughput
- MIG (multi-instance GPU): 一个 A100 可以切分为至多 7 个硬件层面隔离的实例
- HBM2e: 更大的 HBM, 更快的 bandwidth, 更高的 DRAM 使用效率
- structure sparsity: 稀疏运算可以带来 2 倍的算力提升

### A100 技术规格

|                                | A100 80GB PCIe                                                              | A100 80GB SXM                                                                                           |
| ------------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| FP64                           | 9.7 TFLOPS                                                                  | 9.7 TFLOPS                                                                                              |
| FP64 Tensor Core               | 19.5 TFLOPS                                                                 | 19.5 TFLOPS                                                                                             |
| FP32                           | 19.5 TFLOPS                                                                 | 19.5 TFLOPS                                                                                             |
| Tensor Float 32 (TF32)         | 156 TFLOPS \| 312 TFLOPS                                                    | 156 TFLOPS \| 312 TFLOPS*                                                                               |
| BFLOAT16 Tensor Core           | 312 TFLOPS \| 624 TFLOPS*                                                   | 312 TFLOPS \| 624 TFLOPS*                                                                               |
| FP16 Tensor Core               | 312 TFLOPS \| 624 TFLOPS*                                                   | 312 TFLOPS \| 624 TFLOPS*                                                                               |
| INT8 Tensor Core               | 624 TOPS \| 1248 TOPS*                                                      | 624 TOPS \| 1248 TOPS*                                                                                  |
| GPU Memory                     | 80GB HBM2e                                                                  | 80GB HBM2e                                                                                              |
| GPU Memory Bandwidth           | 1,935 GB/s                                                                  | 2,039 GB/s                                                                                              |
| Max Thermal Design Power (TDP) | 300W                                                                        | 400W ***                                                                                                |
| Multi-Instance GPU             | Up to 7 MIGs @ 10GB                                                         | Up to 7 MIGs @ 10GB                                                                                     |
| Form Factor                    | PCIe  <br>Dual-slot air-cooled or single-slot liquid-cooled                 | SXM                                                                                                     |
| Interconnect                   | NVIDIA® NVLink® Bridge  <br>for 2 GPUs: 600 GB/s **  <br>PCIe Gen4: 64 GB/s | NVLink: 600 GB/s  <br>PCIe Gen4: 64 GB/s                                                                |
| Server Options                 | Partner and NVIDIA-Certified Systems™ with 1-8 GPUs                         | NVIDIA HGX™ A100-Partner and NVIDIA-Certified Systems with 4,8, or 16 GPUs NVIDIA DGX™ A100 with 8 GPUs |

## H100

### H100 关键改进

- Hopper 架构
- Tensor Core: 更强的 tensor core
- transformer engine: 加速基于 transformer 架构模型的训练
- NVLink: 900GB/s 的 bandwidth
- 2nd MIG: 支持 multi-tenant, multi-user 使用
- DPX: 基于 DPX 指令集加速动态规划算法

### H100 技术规格

|                                | H100 SXM                                                                                                      | H100 NVL                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| FP64                           | 34 teraFLOPS                                                                                                  | 30 teraFLOPs                                       |
| FP64 Tensor Core               | 67 teraFLOPS                                                                                                  | 60 teraFLOPs                                       |
| FP32                           | 67 teraFLOPS                                                                                                  | 60 teraFLOPs                                       |
| TF32 Tensor Core*              | 989 teraFLOPS                                                                                                 | 835 teraFLOPs                                      |
| BFLOAT16 Tensor Core*          | 1,979 teraFLOPS                                                                                               | 1,671 teraFLOPS                                    |
| FP16 Tensor Core*              | 1,979 teraFLOPS                                                                                               | 1,671 teraFLOPS                                    |
| FP8 Tensor Core*               | 3,958 teraFLOPS                                                                                               | 3,341 teraFLOPS                                    |
| INT8 Tensor Core*              | 3,958 teraFLOPS                                                                                               | 3,341 teraFLOPS                                    |
| GPU Memory                     | 80GB                                                                                                          | 94GB                                               |
| GPU Memory Bandwidth           | 3.35TB/s                                                                                                      | 3.9TB/s                                            |
| Decoders                       | 7 NVDEC  <br>7 JPEG                                                                                           | 7 NVDEC  <br>7 JPEG                                |
| Max Thermal Design Power (TDP) | Up to 700W (configurable)                                                                                     | 350-400W (configurable)                            |
| Multi-Instance GPUs            | Up to 7 MIGS @ 10GB each                                                                                      | Up to 7 MIGS @ 12GB each                           |
| Form Factor                    | SXM                                                                                                           | PCIe  <br>dual-slot air-cooled                     |
| Interconnect                   | NVIDIA NVLink™: 900GB/s  <br>PCIe Gen5: 128GB/s                                                               | NVIDIA NVLink: 600GB/s  <br>PCIe Gen5: 128GB/s     |
| Server Options                 | NVIDIA HGX H100 Partner and NVIDIA-  <br>Certified Systems™ with 4 or 8 GPUs  <br>NVIDIA DGX H100 with 8 GPUs | Partner and NVIDIA-Certified Systems with 1–8 GPUs |
| NVIDIA AI Enterprise           | Add-on                                                                                                        | Included                                           |

## H200

### H200 关键改进

- 更高的 HBM 内存和带宽
- 更高的 LLM inference 速度

### H200 技术规格

|                                | H200 SXM                                                                     | H200 NVL                                                                         |
| ------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| FP64                           | 34 teraFLOPS                                                                 | 30 teraFLOPs                                                                     |
| FP64 Tensor Core               | 67 teraFLOPS                                                                 | 60 teraFLOPs                                                                     |
| FP32                           | 67 teraFLOPS                                                                 | 60 teraFLOPs                                                                     |
| TF32 Tensor Core*              | 989 teraFLOPS                                                                | 835 teraFLOPs                                                                    |
| BFLOAT16 Tensor Core*          | 1,979 teraFLOPS                                                              | 1,671 teraFLOPS                                                                  |
| FP16 Tensor Core*              | 1,979 teraFLOPS                                                              | 1,671 teraFLOPS                                                                  |
| FP8 Tensor Core*               | 3,958 teraFLOPS                                                              | 3,341 teraFLOPS                                                                  |
| INT8 Tensor Core*              | 3,958 teraFLOPS                                                              | 3,341 teraFLOPS                                                                  |
| GPU Memory                     | **141GB**                                                                    | **141GB**                                                                        |
| GPU Memory Bandwidth           | **4.8TB/s**                                                                  | **4.8TB/s**                                                                      |
| Decoders                       | 7 NVDEC  <br>7 JPEG                                                          | 7 NVDEC  <br>7 JPEG                                                              |
| Confidential Computing         | Supported                                                                    | Supported                                                                        |
| Max Thermal Design Power (TDP) | Up to 700W (configurable)                                                    | Up to **600W** (configurable)                                                    |
| Multi-Instance GPUs            | Up to 7 MIGS @ **18GB** each                                                 | Up to 7 MIGS @ **18GB** each                                                     |
| Form Factor                    | SXM                                                                          | PCIe  <br>dual-slot air-cooled                                                   |
| Interconnect                   | NVIDIA NVLink™: 900GB/s <br>PCIe Gen5: 128GB/s                               | 2- or 4-way NVIDIA NVLink bridge: ** 900GB/s**  per GPU<br>PCIe Gen5: 128GB/s    |
| Server Options                 | NVIDIA HGX H200 Partner and NVIDIA-  <br>Certified Systems™ with 4 or 8 GPUs | NVIDIA MGX™ H200 NVL partner and  <br>NVIDIA-Certified Systems with up to 8 GPUs |
| NVIDIA AI Enterprise           | Add-on                                                                       | Included                                                                         |

相比于 H100, H200 升级了 HBM 和 bandwidth

## B200

### B200 关键改进

- blackwell 架构： GPU 之间的通信效率大幅度提升
- Grace CPU: GPU 可以与 Grace CPu 之间达到 900GB/s 的 bidirectional bandwidth
- 5th NVIDIA NVLink: 可以链接 576 块 GPU 来支持计算，NVlink 的带宽可以达到 130TB/s
- RAS engine: 自动识别故障来提高效率
- NVIDIA networking

### B2100 技术规格

system specification 如下

| Specification                       | GB200 NVL72                  | GB200 NVL4                | HGX B200         |
| ----------------------------------- | ---------------------------- | ------------------------- | ---------------- |
| NVIDIA Blackwell GPUs \| Grace CPUs | 72 \| 36                     | 4 \| 2                    | 8 \| 0           |
| CPU Cores                           | 2,592 Arm® Neoverse V2 Cores | 144 Arm Neoverse V2 Cores | -                |
| Total NVFP4 Tensor Core²            | 1,440 \| 720 PFLOPS          | 80 \| 40 PFLOPS           | 144 \| 72 PFLOPS |
| Total FP8/FP6 Tensor Core²          | 720 PFLOPS                   | 40 PFLOPS                 | 72 PFLOPS        |
| Total Fast Memory                   | 31 TB                        | 1.8 TB                    | 1.4 TB           |
| Total Memory Bandwidth              | 576 TB/s                     | 32 TB/s                   | 62 TB/s          |
| Total NVLink Bandwidth              | 130 TB/s                     | 7.2 TB/s                  | 14.4 TB/s        |

individual specification 如下

| Specification                  | GB200 NVL72                                                           | GB200 NVL4                                                | HGX B200                                                         |
| ------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------- |
| FP4 Tensor Core                | 20 PFLOPS                                                             | 20 PFLOPS                                                 | 18 PFLOPS                                                        |
| FP8/FP6 Tensor Core²           | 10 PFLOPS                                                             | 10 PFLOPS                                                 | 9 PFLOPS                                                         |
| INT8 Tensor Core²              | 10 POPS                                                               | 10 POPS                                                   | 9 POPS                                                           |
| FP16/BF16 Tensor Core²         | 5 PFLOPS                                                              | 5 PFLOPS                                                  | 4.5 PFLOPS                                                       |
| TF32 Tensor Core²              | 2.5 PFLOPS                                                            | 2.5 PFLOPS                                                | 2.2 PFLOPS                                                       |
| FP32                           | 80 TFLOPS                                                             | 80 TFLOPS                                                 | 75 TFLOPS                                                        |
| FP64 / FP64 Tensor Core        | 40 TFLOPS                                                             | 40 TFLOPS                                                 | 37 TFLOPS                                                        |
| GPU Memory <br>Bandwidth       | 186 GB HBM3E <br>8 TB/s                                               | 186 GB HBM3E <br>8 TB/s                                   | 180 GB HBM3E <br>7.7 TB/s                                        |
| Multi-Instance GPU (MIG)       | -                                                                     | 7                                                         | -                                                                |
| Decompression Engine           | -                                                                     | Yes                                                       | -                                                                |
| Decoders                       | -                                                                     | 7 NVDEC³ <br>7 nvJPEG                                     | -                                                                |
| Max Thermal Design Power (TDP) | Configurable up to 1,200 W                                            | Configurable up to 1,200 W                                | Configurable up to 1,000 W                                       |
| Interconnect                   | -                                                                     | Fifth-generation NVLink: 1.8 TB/s <br>PCIe Gen5: 128 GB/s | -                                                                |
| Server Options                 | NVIDIA GB200 NVL72 partner and NVIDIA-Certified Systems™ with 72 GPUs | NVIDIA MGX partner and NVIDIA-Certified Systems           | NVIDIA HGX B200 partner and NVIDIA-Certified Systems with 8 GPUs |

## B300

### B300 关键改进

- Blackwell 架构
- AI reasoning inference: 支持 test-time scaling, 对 attention layer 和 FLOPs 都有加速
- HBM3e: 支持更大的 batch size 和 throughput
- ConnectX-8 SuperNIC, 一个 host2 个 ConnectX-8 设备，支持 800Gb/s 的 GPU 之间通信
- Grace-CPU: 更强的表现和带宽
- 5th NVIDIA NVLink: 更高的通信效率

### B3100 技术规格

system specification 如下

|                                   | GB300 NVL72                                                   | HGX B300                                                   |
| --------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------- |
| Blackwell Ultra GPUs\| Grace CPUs | 72 \| 36                                                      | 8 \| 0                                                     |
| CPU Cores                         | 2,592 Arm Neoverse V2 Cores                                   | -                                                          |
| Total FP4 Tensor Core             | 1 1,440 PFLOPS \| 1,080 PFLOPS                                | 144 PFLOPS \| 108 PFLOPS                                   |
| Total FP8/FP6 Tensor Core         | 2 720 PFLOPS                                                  | 72 PFLOPS                                                  |
| Total Fast Memory                 | 37 TB                                                         | 2.1 TB                                                     |
| Total Memory Bandwidth            | 576 TB/s                                                      | 62 TB/s                                                    |
| Total NVLink Switch Bandwidth     | 130 TB/s                                                      | 14.4 TB/s                                                  |

individual specification 如下

|                                | GB300 NVL72                                                   | HGX B300                                                   |
| ------------------------------ | ------------------------------------------------------------- | ---------------------------------------------------------- |
| FP4 Tensor Core                | 20 PFLOPS \| 15 PFLOPS                                        | 18 PFLOPS \| 14 PFLOPS                                     |
| FP8/FP6 Tensor Core2           | 10 PFLOPS                                                     | 9 PFLOPS                                                   |
| INT8 Tensor Core2              | 330 TOPS                                                      | 307 TOPS                                                   |
| FP16/BF16 Tensor Core          | 5 PFLOPS                                                      | 4.5 PLFOPS                                                 |
| TF32 Tensor Core2              | 2.5 PFLOPS                                                    | 2.2 PFLOPS                                                 |
| FP32                           | 80 TFLOPS                                                     | 75 TFLOPS                                                  |
| FP64/FP64 Tensor Core          | 1.3 TFLOPS                                                    | 1.2 TFLOPS                                                 |
| GPU Memory \| Bandwidth        | 279 GB HBM3E \| 8 TB/s                                        | 270 GB HBM3E \| 7.7 TB/s                                   |
| Multi-Instance GPU (MIG)       | 7                                                             | 7                                                          |
| Decompression Engine           | Yes                                                           | Yes                                                        |
| Decoders                       | 7 NVDEC3  <br>7 nvJPEG                                        | 7 NVDEC3  <br>7 nvJPEG                                     |
| Max Thermal Design Power (TDP) | Configurable up to 1,400 W                                    | Configurable up to 1,100 W                                 |
| Interconnect                   | Fifth-Generation NVLink: 1.8 TB/s  <br>PCIe Gen6: 256 GB/s    | Fifth-Generation NVLink: 1.8 TB/s  <br>PCIe Gen6: 256 GB/s |
| Server Options                 | NVIDIA GB300 NVL72 partner and  <br>NVIDIA-Certified Systems™ | NVIDIA HGX B300 partner and  <br>NVIDIA-Certified Systems  |

## References

- [V100 white paper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- [A100](https://www.nvidia.com/en-us/data-center/a100/)
- [Hopper Architecture](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
- [H100](https://www.nvidia.com/en-us/data-center/h100/)
- [H200](https://www.nvidia.com/en-us/data-center/h200/)
- [B200](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [B300](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet)
- [blackwell](https://resources.nvidia.com/en-us-gpu-resources/blackwell-ultra-datasheet?lx=CPwSfP)

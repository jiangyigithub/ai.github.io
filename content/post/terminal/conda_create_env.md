---
title: How to fix http error when creating a new environment.
description: Conda configuration
date: 2023-08-25 00:00:00+0000
---

# Problem description

When we use `conda create --env myenv` to create a new environment, it may throw an exception

```bash
CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/main/osx-64/repodata.json>
```

usually, this occurs when the source channels are not domestic. In this case, we need the change the configuration.

# Solution

1. use `conda info` to view the configuration file:

```bash
conda info
```

focus on the `user config file` and open it.
2. use the following configuration to overwrite the file:

```bash
ssl_verify: false

channels:
 - defaults
show_channel_urls: true
default_channels:
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
 conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
```

---
title: Git authentication error
description: Git clone authentication error.
date: 2025-02-22 10:51:27+0800
tags: 
    - Linux
    - Git
categories:
    - Terminal
---

# 问题描述

在使用 `git clone` 命令时，可能会遇到认证错误。

```bash
> git clone https://github.com/user-name/some-repository.git
正克隆到 'some-repository'...
remote: Invalid username or password.
fatal: 'https://github.com/user-name/some-repository.git/' 鉴权失败
```

# 问题原因&解决方案

本地git配置问题，需要重新认证本地和GitHub的连接。一般是因为本地Git无法和Github的SSH服务器连接。

## 方案1：SSH

本方法将Github视为一个SSH服务器，本机通过SSH连接到Github。该方法主要需要在本地生成SSH密钥，并将其添加到Github的SSH密钥列表中。最后通过SSH的方式进行Git操作

1. 本地生成SSH秘钥

```bash
ssh-keygen
```

默认的秘钥地址是`~/.ssh/id_rsa`，对应的公钥就是`~/.ssh/id_rsa.pub`。如果需要指定秘钥地址，可以使用`-f`选项，对于指定的秘钥地址，需要通过以下命令将秘钥地址加入到搜索列表中.

```bash
# 1. Start the SSH agent
> eval "$(ssh-agent -s)"

# 2. Add your custom key file
> ssh-add path/to/id_rsa
```

2. 将公钥添加到Github的SSH密钥列表中。使用 `cat ~/.ssh/id_rsa.pub` 复制公钥，然后打开`Github->设置->SSH and GPG keys->New SSH key`，将公钥粘贴到`Key`中，然后点击`Add SSH key`。将公钥`Key`中并保存。

3. 本机通过SSH连接到Github，可以通过以下方式验证

```bash
> ssh -T git@github.com
# ssh -vT git@github.com # 查看ssh连接的秘钥文件搜索列表
# ssh -i ~/.ssh/id_rsa git@github.com # 自定义密钥的地址，如果使用默认的密钥地址，则不需要指定密钥地址，如果之前已经加入到搜索列表中，则也不需要指定密钥地址
Hi user-name! You've successfully authenticated, but GitHub does not provide shell access.
```

4. 克隆仓库：在clone的时候选择`SSH`的方式（不是`HTTPS`的方式），即repo的地址为`git@github.com:user-name/some-repository.git`
5. 设置remote repo：`git remote set-url origin git@github.com:user-name/some-repository.git`

通过以上步骤，即可正常使用Git进行操作。

## 方案2：Personal Access Token

该方法通过生成一个Personal Access Token来认证本地和GitHub的连接。该方法需要在Github的设置中生成一个Personal Access Token，然后通过该Token来连接remote repo

1. 生成Personal Access Token. 打开`Github->设置->Developer settings->Personal access tokens->Token(classic)->Generate new token`，在`Note`中输入本机相关信息，在`Expiration`中选择token失效日期，在`Permissions`中选择`repo`（必选，其他可选），然后点击`Generate token`。将生成的`<token>`复制到本地。
2. 克隆仓库：`git clone https://user-name:<token>@github.com/user-name/some-repository.git`
3. 设置remote repo：`git remote set-url origin https://user-name:<token>@github.com/user-name/some-repository.git`

通过以上步骤，即可正常使用Git进行操作。

# 参考资料

- [Error: Permission denied (publickey)
](https://docs.github.com/en/authentication/troubleshooting-ssh/error-permission-denied-publickey)
- [Managing your personal access tokens
](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

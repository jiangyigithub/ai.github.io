---
title: Screen usage
description: Screen usage
date: 2025-02-05 15:14:43+0800
tags: 
    - Linux
categories:
    - Terminal
---

# Introduction

Screen is a terminal multiplexer, which allows you to create multiple sessions in a single terminal window.

# Installation

```bash
sudo apt-get install screen
```

# Usage

```bash
screen [-option] [command]
```

Options:

- `-S`: specify the session name
- `-d`: detach from the session
- `-r`: reattach to the session
- `-c`: execute the command in the session
- `-L`: list all sessions
- `-wipe`: wipe out all sessions
- `-x`: execute the command in the session
- `-p`: specify the port number
- `-m`: specify the mode
- `-t`: specify the title
- `-v`: specify the version
- `-h`: display help information
- `-v`: display version information
- `-q`: quit the session

# Reference

- [Screen usage](https://www.gnu.org/software/screen/manual/screen.html)

---
title: 利用hugo和github搭建个人网址
description: 本文介绍了利用go，hugo，github搭建个人网址的过程
date: 2026-04-01 17:44:26+08:00
math: true
tags: 
    - scaling
categories:
    - Infra
---

## 安装go和hugo
https://gentleostrich.github.io/posts/windows-11-%E5%88%9B%E5%BB%BA-hugo-%E7%BD%91%E7%AB%99/

官方教程不允许使用 CMD、Windows PowerShell 执行各个命令，建议使用 PowerShell 执行命令。Anaconda Powershell Prompt 应该是基于 PowerShell 开发的，因此，本文的命令均在 Anaconda Powershell Prompt 中执行。

## 安装 Hugo

安装 Hugo 前需要安装 Git、Go、Dart Sass。

* 安装 [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

  无论是克隆 Hugo 的 GitHub 仓库，还是基于 Github Page 发布网站，都需要使用 Git。

* 安装 [Go](https://go.dev/doc/install)

  Hugo 是基于 Go 语言开发，Go 作为一种高效的编程语言，使得 Hugo 在速度上远超 Hexo 等静态网站搭建技术。

* 安装 [Dart Sass](https://gohugo.io/functions/css/sass/#dart-sass)

  Dart Sass 是 Hugo 开发的网页渲染插件，可以通过 Scoop 和 Chocolatey 两个 Windows 安装器安装，这里选择使用 Scoop 安装。

  * 安装 [Scoop](https://scoop.sh/#/)

    首先 cd 到 C:\ 路径，在 C:\ 路径下执行下述命令。

    ```shell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
    ```

    按照上述官方命令安装时出现了 PowerShell 禁止运行脚本的报错，使用 [以下命令](https://blog.csdn.net/tongxin_tongmeng/article/details/128150906) 替代上面的第一条命令解决该问题。

    ```shell
    Set-ExecutionPolicy RemoteSigned -Scope Process
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    Set-ExecutionPolicy RemoteSigned -Scope LocalMachine
    ```

  安装 Scoop 后，可以执行下述命令安装 Dart Sass。

  ```shell
  scoop install sass
  ```

安装 Git、Go、Dart Sass 后

1. 在 [Hugo GitHub 发布页](https://github.com/gohugoio/hugo/releases/tag/v0.154.5) 下载并解压 [hugo_0.154.5_windows-amd64.zip](https://github.com/gohugoio/hugo/releases/download/v0.154.5/hugo_0.154.5_windows-amd64.zip) 。
2. 将解压得到的文件 hugo.exe 移动到安装路径，这里选择的安装路径是 D:\Program Files\hugo，并将该路径添加到系统路径变量中。

此时，重新打开 PowerShell 并执行 `hugo version`，若输出 Hugo 的版本号等信息，则表明 Hugo 安装成功。

## 网站创建

成功安装 Hugo 后，就可以创建第一个网站了。

```shell
hugo new site quickstart
cd quickstart
git init
git submodule add https://github.com/theNewDynamic/gohugo-theme-ananke.git themes/ananke
echo "theme = 'ananke'" >> hugo.toml
hugo server
```

* 报错 1：

  ```shell
  ERROR command error: failed to load config: "D:\Program Files\hugo\quickstart\hugo.toml:4:2": unmarshal failed: toml: expected character =
  ```

  直接将 `echo "theme = 'ananke'" >> hugo.toml` 复制到 PowerShell 中执行会出现格式错误，修改 hugo.toml 


* 报错 2：

  ```shell
  ERROR error building site: render: [en v1.0.0 guest] failed to render pages: render of "/tags" failed: "D:\Program Files\hugo\quickstart\themes\ananke\layouts\baseof.html:26:15": execute of template failed: template: taxonomy.html:26:15: executing "taxonomy.html" at <partials.Include>: error calling Include: "D:\Program Files\hugo\quickstart\themes\ananke\layouts\_partials\site-style.html:2:32": execute of template failed: template: _partials/site-style.html:2:32: executing "_partials/site-style.html" at <.RelPermalink>: error calling RelPermalink: TOCSS: failed to transform "/ananke/css/main.css" (text/css). Check your Hugo installation; you need the extended version to build SCSS/SASS with transpiler set to 'libsass'.: this feature is not available in your current Hugo version, see https://goo.gl/YMrWcn for more information
  ```

  这是因为通过 [hugo_0.154.5_windows-amd64.zip](https://github.com/gohugoio/hugo/releases/download/v0.154.5/hugo_0.154.5_windows-amd64.zip) 安装的不是 extended version，不能对 anake 主题进行渲染。执行命令 `scoop install hugo-extended` 直接从 scoop 中安装 hugo-extended 并删除系统路径中的 hugo 路径即可。

## 网站配置

通过目录下的 hugo.toml 配置网站，其中的 baseURL 设置为生产网站。

```toml
baseURL = 'https://example.org/'
languageCode = 'en-us'
title = 'My New Hugo Site'
theme = 'ananke'
```

## 内容添加

向网站中添加一个新文章：

```shell
hugo new content content/posts/my-first-post.md
```

在 content/posts 目录下出现了 my-first-post.md 文件，下述是该文件中的内容。其中 draft=true 表示这个文章不会被发表到网站上。

```markdown
+++
title = 'My First Post'
date = 2024-01-14T07:07:07+01:00
draft = true
+++
```

通过下述命令查看含有 draft 文章的网站，当决定发表该文章后，将 draft 改为 false。

```shell
hugo server --buildDrafts
hugo server -D
```

## 网站发布

网站发布指的是 Hugo 在 public 目录下创建静态网站所需的全部文件（包括 HTML 文件、CSS 文件、Javascript 文件、图片得等等），命令如下：

```shell
hugo
```

## 网站部署

基于 GitHub Page 部署网站。

1. 新建项目，项目名称为：\<username\>.github.io。这意味着这个项目是一个网站，每次 push 后都会进行 action 操作进行网站部署。

2. 将本地 Hugo 仓库与 GitHub 项目仓库进行 remote 关联，代码如下：

   ```shell
   git remote add origin https://github.com/GentleOstrich/gentleostrich.github.io.git
   git branch -M main
   git push -u origin main
   ```

   报错：

   ```shell
   git push -u origin mainerror: src refspec main does not match any
   error: failed to push some refs to 'https://github.com/GentleOstrich/gentleostrich.github.io.git'
   ```

   原因是最开始本地项目从未 commit 过，所以没有不存在分支 main。执行下述命令即可解决：

   ```shell
   git commit -m "Initial commit"
   git push -u origin main
   ```

3. 设置图片缓存位置

   在 hugo.toml 文件中添加下述内容：

   ```shell
   [caches]
     [caches.images]
       dir = ':cacheDir/images'
   ```

4. 设置 action 文件

   action 文件的作用是指导 GitHub 如何部署网站的，首先创建相关 hugo.yaml 文件：

   ```shell
   mkdir -p .github/workflows
   touch .github/workflows/hugo.yaml
   ```

   接下来在 hugo.yaml 文件中加入下述内容：

   ```yaml
   name: Build and deploy
   on:
     push:
       branches:
         - main
     workflow_dispatch:
   permissions:
     contents: read
     pages: write
     id-token: write
   concurrency:
     group: pages
     cancel-in-progress: false
   defaults:
     run:
       shell: bash
   jobs:
     build:
       runs-on: ubuntu-latest
       env:
         DART_SASS_VERSION: 1.97.2
         GO_VERSION: 1.25.5
         HUGO_VERSION: 0.154.4
         NODE_VERSION: 24.12.0
         TZ: Europe/Oslo
       steps:
         - name: Checkout
           uses: actions/checkout@v6
           with:
             submodules: recursive
             fetch-depth: 0
         - name: Setup Go
           uses: actions/setup-go@v6
           with:
             go-version: ${{ env.GO_VERSION }}
             cache: false
         - name: Setup Node.js
           uses: actions/setup-node@v6
           with:
             node-version: ${{ env.NODE_VERSION }}
         - name: Setup Pages
           id: pages
           uses: actions/configure-pages@v5
         - name: Create directory for user-specific executable files
           run: |
             mkdir -p "${HOME}/.local"
         - name: Install Dart Sass
           run: |
             curl -sLJO "https://github.com/sass/dart-sass/releases/download/${DART_SASS_VERSION}/dart-sass-${DART_SASS_VERSION}-linux-x64.tar.gz"
             tar -C "${HOME}/.local" -xf "dart-sass-${DART_SASS_VERSION}-linux-x64.tar.gz"
             rm "dart-sass-${DART_SASS_VERSION}-linux-x64.tar.gz"
             echo "${HOME}/.local/dart-sass" >> "${GITHUB_PATH}"
         - name: Install Hugo
           run: |
             curl -sLJO "https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.tar.gz"
             mkdir "${HOME}/.local/hugo"
             tar -C "${HOME}/.local/hugo" -xf "hugo_extended_${HUGO_VERSION}_linux-amd64.tar.gz"
             rm "hugo_extended_${HUGO_VERSION}_linux-amd64.tar.gz"
             echo "${HOME}/.local/hugo" >> "${GITHUB_PATH}"
         - name: Verify installations
           run: |
             echo "Dart Sass: $(sass --version)"
             echo "Go: $(go version)"
             echo "Hugo: $(hugo version)"
             echo "Node.js: $(node --version)"
         - name: Install Node.js dependencies
           run: |
             [[ -f package-lock.json || -f npm-shrinkwrap.json ]] && npm ci || true
         - name: Configure Git
           run: |
             git config core.quotepath false
         - name: Cache restore
           id: cache-restore
           uses: actions/cache/restore@v5
           with:
             path: ${{ runner.temp }}/hugo_cache
             key: hugo-${{ github.run_id }}
             restore-keys:
               hugo-
         - name: Build the site
           run: |
             hugo \
               --gc \
               --minify \
               --baseURL "${{ steps.pages.outputs.base_url }}/" \
               --cacheDir "${{ runner.temp }}/hugo_cache"
         - name: Cache save
           id: cache-save
           uses: actions/cache/save@v5
           with:
             path: ${{ runner.temp }}/hugo_cache
             key: ${{ steps.cache-restore.outputs.cache-primary-key }}
         - name: Upload artifact
           uses: actions/upload-pages-artifact@v4
           with:
             path: ./public
     deploy:
       environment:
         name: github-pages
         url: ${{ steps.deployment.outputs.page_url }}
       runs-on: ubuntu-latest
       needs: build
       steps:
         - name: Deploy to GitHub Pages
           id: deployment
           uses: actions/deploy-pages@v4
   ```

至此，等待 GitHub 项目主页中 Action 显示绿色对勾后，就可以通过网址 \<username\>.github.io 访问网站啦（可能显示绿色对勾后立刻访问网站会出现 403 错误，只需要再等待一会儿就可以正常访问了）

## [shortcode](https://gohugo.io/content-management/shortcodes/)

可用于内嵌视频、图片等元素，分为 embedded custom inline 三类。

插入一个 Youtube 视频（是一个 embedded）：{{< youtube YQHsXMglC9A >}}`

插入日期（是一个 inline）：今天是 {{< date.inline ":date_medium" >}}{{- now | time.Format (.Get 0) -}}{{< /date.inline >}}。

插入二维码：{{< qr >}}https://gentleostrich.github.io{{< /qr >}}


## 设置github actions部署 workflow
https://github.com/jiangyigithub/ai.github.io/actions
`action workflow`
.github\workflows\deploy.yml

## 设置github page的
`hugo github pages`
https://github.com/jiangyigithub/ai.github.io/settings/pages

## 更新文章
content\post

## 个人网址配置
`github.io website`
baseurl = "https://github.com/jiangyigithub/ai.github.io"
config\_default\config.toml
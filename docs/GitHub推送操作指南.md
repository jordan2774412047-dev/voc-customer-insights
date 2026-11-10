# Git 推送到 GitHub 操作指南

适用项目：`voc-customer-insights`  
本地目录：`$env:USERPROFILE\Desktop\VOC助手原型`  
GitHub 仓库：<https://github.com/jordan2774412047-dev/voc-customer-insights>  
默认分支：`main`

> 当前项目已经完成 Git 初始化、首次合并和首次推送，远程名为 `origin`。日常更新从“日常提交和推送”开始即可。

---

## 1. 日常提交和推送

### 1.1 打开项目目录

打开 PowerShell，运行：

```powershell
cd "$env:USERPROFILE\Desktop\VOC助手原型"
```

如果项目以后移动了，改成新的项目目录。

### 1.2 查看状态

```powershell
git status --short --branch
```

正常情况下会看到当前分支为 `main`。修改过的文件会列在状态输出中。

### 1.3 检查准备提交的改动

先查看文件列表和差异，确认只包含本次要保存的项目文件：

```powershell
git status
git diff
```

把所有已检查的改动加入暂存区：

```powershell
git add --all
```

再核对暂存区内容：

```powershell
git diff --cached --stat
git diff --cached
```

如果误暂存某个文件，可从暂存区移除，但保留本地文件：

```powershell
git restore --staged <文件路径>
```

### 1.4 创建本地提交

```powershell
git commit -m "描述这次修改"
```

提交说明建议具体，例如：

```powershell
git commit -m "docs: add GitHub push guide"
```

### 1.5 推送到 GitHub

```powershell
git push
```

首次推送并设置上游分支时使用：

```powershell
git push -u origin main
```

之后 `main` 已跟踪 `origin/main`，通常只需要运行 `git push`。

### 1.6 确认推送成功

```powershell
git status --short --branch
git log -1 --oneline --decorate
```

同步成功时，状态一般显示 `main...origin/main`，且没有未提交的文件变更。也可以刷新 GitHub 仓库页面检查最新提交。

---

## 2. 首次把已有本地仓库连接到 GitHub

本项目已完成此流程，新克隆或新项目才需要参考本节。

### 2.1 创建 GitHub 仓库

在 GitHub 创建仓库并复制 HTTPS 或 SSH 地址。若本地已经有 README、许可证或 `.gitignore`，创建远程仓库时最好不要再次初始化这些文件，以免本地与远程出现两套独立提交历史。

### 2.2 添加远程地址

HTTPS 示例：

```powershell
git remote add origin https://github.com/用户名/仓库名.git
git remote -v
```

如果提示 `remote origin already exists`，检查并更新地址：

```powershell
git remote set-url origin https://github.com/用户名/仓库名.git
git remote -v
```

### 2.3 首次推送

```powershell
git push -u origin main
```

GitHub 官方说明了从本地仓库添加远程地址并推送的完整步骤：<https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github>。

---

## 3. 身份验证

使用 HTTPS 推送时，Git Credential Manager 可能会打开浏览器要求登录或授权。按提示在浏览器完成授权后返回终端即可。

- GitHub 网页账号密码不能直接用作 Git HTTPS 密码。
- 不要把 Personal Access Token、密码或密钥写进命令、代码、README 或聊天消息。
- 若希望使用 SSH，请先在 GitHub 账户配置 SSH 公钥，再将远程地址改成 `git@github.com:用户名/仓库名.git`。

认证官方说明：<https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github>。

---

## 4. 常见问题

### 4.1 `remote origin already exists`

本地已经配置了名为 `origin` 的远程地址。不要重复添加，改用：

```powershell
git remote -v
git remote set-url origin https://github.com/用户名/仓库名.git
```

### 4.2 `non-fast-forward` 或 `fetch first`

远程分支比本地多提交。先获取并合并远程更新，再推送：

```powershell
git pull --no-rebase origin main
git push
```

如果出现 `unrelated histories`，说明本地和远程各自初始化过。确认需要保留两边内容后再运行：

```powershell
git pull origin main --allow-unrelated-histories --no-rebase
```

处理完冲突后，检查文件并完成合并提交：

```powershell
git status
git add <已解决的文件>
git commit
git push -u origin main
```

不要用 `git push --force` 处理普通推送失败，它可能覆盖远程提交。

### 4.3 `CONFLICT (add/add)`

本地与远程都新建了同名文件，最常见的是 README。打开冲突文件，删除 `<<<<<<<`、`=======`、`>>>>>>>` 标记，整理成一份完整内容，然后：

```powershell
git add README.md
git commit
git push
```

在合并提交完成前，推送通常会继续被拒绝。

### 4.4 推送前漏了 `git add` 或 `git commit`

`git push` 只发送已经提交的 Git 历史，不会发送尚未提交的文件修改。回到日常流程执行 `git add`、`git commit`，再推送。

### 4.5 提交了不该上传的文件

先停止推送，并检查：

```powershell
git status --short
git diff --cached --stat
git ls-files
```

本项目 `.gitignore` 会排除原始数据 `data/raw/`、处理数据、模型权重和运行缓存。不要绕过这些规则上传第三方评论语料、未经授权的数据或大体积权重。

如果密钥已经推送到远端，仅删除本地文件并再次推送并不能让历史中的密钥失效；应立即轮换密钥，并按 GitHub 的敏感数据移除指南处理仓库历史。

---

## 5. 本项目的数据提交范围

本仓库保留源码、训练脚本、配置、文档、评估报告、标签模板和合成测试样例。原始评论数据、处理后语料与模型权重不提交，原因是数据许可、隐私、仓库体积和模型文件管理方式等考虑。

推送前可快速复查：

```powershell
git status --short --branch
git diff --cached --check
git ls-files
```

如果意外发现 `data/raw/`、`data/processed/` 或 `models/` 下的文件出现在 `git ls-files` 输出中，先暂停推送并检查 `.gitignore` 与暂存区。

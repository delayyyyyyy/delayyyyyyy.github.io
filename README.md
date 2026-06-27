# 我的学习旅程

基于 Material for MkDocs 的个人学习记录网站。

## 1. 修改个人信息

GitHub 用户名和显示昵称已经设置为 `delayyyyyyy`。你只需把
`your-email@example.com` 替换成自己的邮箱；不想公开邮箱也可以删除该行。

本站使用 GitHub 个人主页仓库 `delayyyyyyy.github.io`。

## 2. 本地预览

使用现有的 Python 3.9 Conda 环境运行：

```powershell
C:\Users\admin\.conda\envs\py39\python.exe -m pip install -r requirements.txt
C:\Users\admin\.conda\envs\py39\python.exe -m mkdocs serve
```

浏览器打开 `http://127.0.0.1:8000`。

## 3. 发布到 GitHub Pages

1. 在 GitHub 新建公开仓库 `delayyyyyyy.github.io`，不要勾选自动创建 README。
2. 把这个目录中的文件提交并推送到仓库的 `main` 分支。
3. 打开仓库的 **Settings → Pages**。
4. 在 **Build and deployment → Source** 中选择 **GitHub Actions**。
5. 推送后等待 `Deploy MkDocs to GitHub Pages` 工作流完成。

网站地址将是：

`https://delayyyyyyy.github.io/`

## 4. 评论与点赞

本站已经配置 Giscus，使用 GitHub Discussions 保存评论，访客需使用 GitHub
账号登录。

仓库的 Discussions 已启用。还需要访问
<https://github.com/apps/giscus>，把 Giscus GitHub App 安装到
`delayyyyyyy.github.io` 仓库。

在文章开头的 YAML 区域加入 `comments: true` 即可显示评论区。Giscus
评论区中的表情反应可作为点赞。若以后需要“无需 GitHub 登录的独立点赞按钮”，
可以再接入 Supabase。

## 5. 发布新记录

复制 `docs/journal/first-note.md`，修改文件名、日期和正文，然后把新文章添加到 `mkdocs.yml` 的 `nav` 中。

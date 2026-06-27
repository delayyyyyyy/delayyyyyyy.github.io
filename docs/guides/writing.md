---
date: 2026-06-27
comments: true
tags:
  - 网站使用
  - Markdown
---

# 如何在这个网站发布新文章

这个网站使用 MkDocs Material。每篇文章都是仓库 `docs/` 目录中的一个
Markdown 文件，文件扩展名为 `.md`。

## 一、文章放在哪里

建议按照内容分类保存：

```text
docs/
├─ journal/          # 日常学习记录
├─ papers/           # 论文阅读
├─ code-reading/     # 源码阅读
├─ guides/           # 网站使用指南
├─ projects.md       # 项目记录
└─ about.md          # 个人介绍
```

文件名建议使用简短的英文小写单词，并用连字符连接。例如：

```text
docs/papers/attention-is-all-you-need.md
docs/code-reading/pytorch-dataloader.md
```

## 二、在 GitHub 网页中新建文章

1. 打开
   [网站仓库](https://github.com/delayyyyyyy/delayyyyyyy.github.io)。
2. 进入 `docs`，再进入准备保存文章的分类目录。
3. 点击 **Add file → Create new file**。
4. 输入文件名，例如 `attention-is-all-you-need.md`。
5. 粘贴下面的模板并填写正文。
6. 点击页面底部的 **Commit changes** 保存。

如果所需目录还不存在，可以在文件名中直接填写完整路径，例如：

```text
papers/attention-is-all-you-need.md
```

## 三、论文阅读模板

````markdown
---
date: 2026-06-27
comments: true
tags:
  - 论文阅读
  - Transformer
---

# Attention Is All You Need

## 论文信息

- **作者**：
- **发表时间**：
- **论文链接**：
- **代码链接**：

## 研究问题

这篇论文想解决什么问题？过去的方法有什么不足？

## 核心方法

用自己的话说明模型结构、关键公式和重要思想。

## 实验与结论

作者做了哪些实验？结果说明了什么？

## 我的理解

记录真正理解的内容，不必照抄论文摘要。

## 尚未解决的问题

- [ ] 问题一
- [ ] 问题二
````

## 四、源码阅读模板

````markdown
---
date: 2026-06-27
comments: true
tags:
  - 源码阅读
  - PyTorch
---

# PyTorch DataLoader 源码阅读

## 阅读目标

我为什么阅读这部分代码？想弄清楚什么？

## 代码入口

- **仓库**：
- **文件路径**：
- **类或函数**：

## 调用流程

```text
DataLoader
  → _BaseDataLoaderIter
  → _MultiProcessingDataLoaderIter
```

## 关键代码

```python
def example():
    pass
```

## 我的理解

用自己的话解释关键代码、数据流和设计原因。

## 尚未解决的问题

- [ ] 问题一
- [ ] 问题二
````

`comments: true` 会在文章底部显示评论与点赞区域。`tags` 可以填写多个，
用于标记文章主题。

## 五、把文章加入导航栏

创建文章后，打开仓库根目录中的 `mkdocs.yml`，找到 `nav:`。

假设文章路径是：

```text
docs/papers/attention-is-all-you-need.md
```

在 `nav:` 中加入：

```yaml
nav:
  - 首页: index.md

  - 学习记录:
      - 总览: journal/index.md
      - 第一篇记录: journal/first-note.md

  - 论文阅读:
      - Attention Is All You Need: papers/attention-is-all-you-need.md

  - 源码阅读:
      - PyTorch DataLoader: code-reading/pytorch-dataloader.md

  - 项目: projects.md
  - 写作指南: guides/writing.md
  - 关于我: about.md
```

路径从 `docs/` 后面开始写，因此不要在导航路径中重复填写 `docs/`。

!!! warning "注意 YAML 缩进"

    `mkdocs.yml` 必须使用空格缩进，不能使用 Tab。同一层级保持相同缩进，
    冒号后保留一个空格。

## 六、等待自动发布

提交文章和 `mkdocs.yml` 后，GitHub Actions 会自动构建网站。

可以在仓库的 **Actions** 页面查看进度。工作流显示绿色勾号后，刷新
[网站首页](https://delayyyyyyy.github.io/) 即可看到更新。通常需要几十秒。

## 七、发布前自查

- [ ] 文件位于 `docs/` 目录中
- [ ] 文件扩展名是 `.md`
- [ ] 日期和标签已经填写
- [ ] 需要评论区时设置了 `comments: true`
- [ ] 已在 `mkdocs.yml` 的 `nav` 中添加文章
- [ ] YAML 缩进正确

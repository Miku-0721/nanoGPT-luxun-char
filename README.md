# 基于 nanoGPT 的中文语料字符级语言模型复现

> 本项目基于 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)，将训练语料替换为中文文本（《鲁迅全集》），用于学习 Transformer / GPT 的基本原理与深度学习实验全流程。

## 项目简介

本项目基于 nanoGPT 复现了一个字符级 GPT 语言模型，
将原本的英文莎士比亚语料替换为《鲁迅全集》，从数据预处理、模型训练到文本生成，
完整走通了一遍语言模型的实验流程，用于深入理解 Transformer 内部机制。

## 背景与动机

为了理解 LLM 的原理、attention 机制、Transformer 架构、练习完整的深度学习实验流程。

## 环境与复现方式

**环境**
- Python: < 3.12.13 >
- PyTorch: < 2.13.0 + cu130 >
- Device: < NVIDIA RTX 5070 Laptop >
- OS: < Windows 11 >

**复现步骤**

```bash
# 1. 克隆本仓库
git clone https://github.com/Miku-0721/nanoGPT-luxun-char.git
cd nanoGPT-luxun-char

# 2. 创建虚拟环境
conda create -n nanogpt python=3.12
conda activate nanogpt
pip install torch numpy transformers datasets tiktoken wandb tqdm

# 3. 数据预处理
python data/LuXun_char/prepare.py

# 4. 开始训练
python train.py config/train_LuXun_char.py --compile=False

# 5. 生成文本
python sample.py --out_dir=out-luxun-char

> 如果没有 NVIDIA 显卡，在第 4 步命令末尾加上 `--device=cpu` 即可用 CPU 训练。

```

## 数据说明

- 数据来源：[hankinghu/literture books](https://github.com/hankinghu/literature-books/blob/master/%E9%B2%81%E8%BF%85%E5%85%A8%E9%9B%86.txt)
- 数据规模：约 580 KB / 18 万字符
- 分词方式：字符级分词，词表大小 3382
- 训练/验证集划分比例：9:1

## 实验记录

| 实验编号 |        改动内容        | n_layer | n_head | n_embd | 训练数据 | train loss | val loss | 备注 |

| exp-01 | 英文莎士比亚，官方默认配置 |   6   |   6   |   384   | Shakespeare (char) | 0.82 | 1.58 | 明显过拟合，数据集较小属预期现象 |
| exp-02 |      替换为中文语料      |   6   |   6   |   384   |     LuXun (char)   | 0.08 | 7.38 | 明显过拟合，数据集较小属预期现象 |
| exp-03 |        调整参数         |   4   |   4   |   512   |     LuXun (char)   | 2.97 | 5.26 | 过拟合现象有所减轻，说明更改生效 |

### exp-02: 中文语料复现

**改动**：将莎士比亚文选改为鲁迅全集

**结果**：明显过拟合，train loss 趋近于 0

**分析**：词表更大、字符组合更复杂，且数据量太小，模型几乎是在“死记硬背”，生成的内容在原文中几乎都可以找到

### exp-03: 调整超参数

**改动**：将层数和注意力头数减少，同时增加维数

**结果**：过拟合现象有所减轻

**分析**：模型不再机械地“背诵”原文，但仍没有学会足够丰富的语言分布，只学会了"记住片段、重复片段"

## 踩坑记录

| 问题 | 原因 | 解决方法 |

| PowerShell 里 `conda activate` 报错，提示无法加载模块 | conda 未对 PowerShell 做初始化 | 在 Anaconda Prompt 里运行 `conda init powershell` |
| `torch.compile` 报 `UnicodeDecodeError: 'gbk' codec can't decode...` | Windows 默认 GBK 编码，与 PyTorch 内部模板文件的 UTF-8 编码冲突 | 训练时加 `--compile=False` |
| <!-- 你自己遇到的其他问题 --> | | |

## 生成效果展示

**英文生成**

```
LEONTES:
True, why, then, then, Hermioner
Where is the armour of hath temperate,
Lest as hours, to hear his hell.

ANGELO:
Now fair good man I have proved!

ISABELLA:
I would take my true heart.

ANGELO:
I am at the present of it.

ISABELLA:
I do not to scarce already: 'tis a world prize,
The valoner than a battery of the heir of parliaments
When men's favour.

```

**中文生成**

```
他很寂静，回到了一支毫不变了，而且很不安。他们第四乎要紧走。
他的时候，自然而且不是他走了。但又是的时候，他又很觉得几回可以他的，却常觉得不觉了。
“这时候，……。”

他又被人不是异意外是在他放到了。
“他，也还在她们这里去，却害怕要想，你自己觉得胜利害人，可怜的。”

“我在我的是非常的，只是自己觉得不知道和宏儿子。”她被他自己的说

“没有。”
他不了。我已经出来，我也如意外，我的话，在说起来，前那里的话，我自己造反对起来，还是这样。

“我有回，能将是这时，叫我造反而我的。”
我家里去就是怎样的，那中，仿佛听说，我的死了。

“你是因为这一个不如此的。这所以为我的。不是什么，可实是我后来的，是我可以为现在是向我这也很多了；当。你还没有这日，仿佛并没有。我的时候，我这是那时常常常常常不愿意思想到这样。然而我家，她面前，但我的 人，立即使我们在，也还有也就是已经如此而且须在是不知道是我的生了性，她的厌，也只是不但现在这样的，也并且很活下是我的，便是祥林嫂。不愿意。但是这虚开；我希望我有我所料到了。我是今还更使我们知道的。我所谓我的，我只是这在，而这里，先我已经改变

```

**更改参数后的中文生成**

```
“那可……“

“他慌忙，已经情，“也没有问。那里去了。你们！”

“这一点头发了，你们这不知道，便是做官不能够……”

“那里不知道么？”

“这不要去，”

“是…”

““这这是已经懂得。”阔亭恍然还是大声。我忍不合上是的声说。”

“先生说。

“你的。”他却还是………”

“那里同时的。
“我没有什么，就是那是我也没有这铁一人没有什么。”
“还是一个人……………”
“不知道：那一直是第二十样地问。”
“那么，“你自己没有………”

“好，都是做办的。”马，”
“我就是。”
“我和我已经现在后来的之后来的时候，仰而且何况在将你一切已经用伊在这一无反而且发生气愈高兴的神情。

```

### 生成效果点评

- 对话体格式完全掌握：引号、换行、"某某说"这种鲁迅小说里大量出现的对话结构，模拟得很像
- 人物名字用对了地方：羿、四铭、阔亭、七斤嫂、祥林嫂、墨子、公输般，这些都是鲁迅小说里真实出现的角色名，说明模型确实记住了具体篇目的内容
- 语气词和口语化表达很到位：哼、唉、哈哈哈哈、…… 这类鲁迅笔下常见的语气和标点习惯，模仿得很像

## 模型结构简述

`model.py` 里的几个核心类是层层嵌套的关系：

- **`LayerNorm`**：把一批数据的数值拉回稳定范围（均值接近 0、方差接近 1），防止训练过程中数值忽大忽小导致不稳定。
- **`CausalSelfAttention`**：Transformer 最核心的部分，让每个字符"回头看"它前面出现过的所有字符，并根据相关性分配不同的关注权重。"Causal（因果）"指只能看前面、不能看后面——因为要预测下一个字符，看到后面的内容就是作弊了。
- **`MLP`**：一个两层全连接网络，对每个位置单独做非线性变换，让模型对 attention 提取到的信息做进一步加工。
- **`Block`**：把上述三者组装成"一层"结构，顺序是 `LayerNorm → CausalSelfAttention → 残差连接 → LayerNorm → MLP → 残差连接`。配置里的 `n_layer` 决定这个 Block 要堆叠几次。
- **`GPTConfig`**：纯配置类，打包 `n_layer`（层数）、`n_head`（每层的注意力头数）、`n_embd`（每个字符的向量维度）等超参数，供 `GPT` 类初始化时读取。
- **`GPT`**：最顶层的类，把整个模型串起来：输入层（字符 id 转向量 + 位置编码）→ 若干个 `Block` 依次处理 → 最后一层 `LayerNorm` → 输出层 `lm_head`（把向量映射回词表大小的概率分布，预测下一个字符）。

本项目实际训练时用的配置（对比英文基线做了缩小，以缓解过拟合）：

| 参数 | 英文基线 | 中文最终版本 |
|---|---|---|
| n_layer | 6 | 4 |
| n_head | 6 | 4 |
| n_embd | 384 | 512 |
| dropout | 0.2 | 0.3 |

## 反思与后续想法

- 如果有更多算力，想尝试更大的模型 / 更多数据
- 想进一步了解 BPE 分词相比字符级分词的优劣
- 想尝试给模型加入某种条件控制（如按某种风格生成）

## 参考资料

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin/Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [3Blue1Brown/《Transformers, the tech behind LLMs》](https://www.youtube.com/watch?v=wjZofJX0v4M)
- [李沐/《Transformer论文逐段精读》](https://www.bilibili.com/video/BV1pu411o7BE)
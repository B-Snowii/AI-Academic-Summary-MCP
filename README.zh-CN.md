[![Language](https://img.shields.io/badge/Language-English-blue)](README.md)
[![语言](https://img.shields.io/badge/语言-中文-red)](README.zh-CN.md)

# AI Academic Summary MCP

## 📋 项目概述

一个基于MCP（Model Context Protocol）的智能学术论文摘要系统，专为研究团队设计，能够快速生成高质量的25字学术摘要。

### 💡 解决的问题
**传统挑战**: 每周需要阅读15-20篇论文，在3-4天内完成所有摘要，根据同事反馈修改，确保摘要符合团队学术标准。

**AI解决方案**: AI系统快速生成初稿，MCP工作流确保效率，JSON配置自动学习团队标准，基于反馈持续质量改进。

## 🎯 核心成果
- **时间成本**：40+ 小时/周 → ≤2 小时/周（**95%效率提升**）
- **质量保障**：方法/主题准确、风格一致
- **可复用性**：MCP流程 + 历史修改数据
- **团队协作**：标准化输出，便于同事修改和反馈

## 🏗️ 系统架构
```
PDF上传 → 在线搜索 → 多模型推理 → CoT分析 → 约束检查 → 风格重写 → 自评打分 → 最终摘要
     ↓         ↓           ↓          ↓         ↓          ↓          ↓          ↓
自动解析 → 实时信息 → AI+人类协作 → 4步逻辑 → 质量控制 → 学术语调 → 质量评分 → 25词输出
```

### 🎯 核心功能
- **智能摘要生成**: 基于多模型协作，生成符合学术标准的25字摘要
- **实时在线搜索**: 集成在线搜索功能，补充论文的实时学术信息
- **4步推理框架**: 采用Chain of Thought推理，确保摘要质量和准确性
- **团队协作优化**: 支持团队反馈学习，持续改进摘要质量

**模型**：Deepseek‑R1、Llama‑3.1  
**相关仓库**：https://github.com/B-Snowii/Research-Paper-Summary-Collection

## 📊 竞争分析

| 功能特性 | 我们的系统 | ChatPaper | AI Summarizer |
|---------|------------|-----------|---------------|
| **在线搜索** | ✅ 实时 | ❌ 静态 | ❌ 无 |
| **多模型** | ✅ AI+人工 | ❌ 单一 | ❌ 单一 |
| **CoT推理** | ✅ 4步 | ❌ 基础 | ❌ 无 |
| **参数控制** | ✅ 高级 | ❌ 固定 | ❌ 无 |

## 📝 摘要示例

示例1: This paper examines how belief vs taste drivers shape early-stage ESG collaboration, via randomized experiments with founders and VCs; methods: randomized assignment and survey-based measures.

示例2: This paper uses a calibrated life-cycle model to value reductions in health risks and quantify insurance, financial, and fiscal impacts; methods: structural modeling with parameter calibration.

示例3: This paper exploits staggered adoption of hospital pay-transparency laws to study effects on patient satisfaction; methods: panel data analysis with staggered policy timing and fixed effects.

## 🧠 CoT推理框架

系统采用结构化的4步Chain of Thought推理过程：

### 第1步: 主题与方法识别
- 提取核心研究主题和方法论
- 识别关键技术方法和创新点

### 第2步: 内容分析与在线增强
- 结合在线搜索分析论文内容
- 补充实时学术信息

### 第3步: 约束验证
- 验证25词长度要求
- 确保学术语调和术语准确性
- 验证方法/主题一致性

### 第4步: 风格优化与自评分
- 应用学术写作标准
- 自我评估摘要质量
- 生成最终优化输出

## 🚀 本地安装

**克隆仓库：**
```bash
git clone https://github.com/B-Snowii/AI-Academic-Summary-MCP.git
cd AI-Academic-Summary-MCP
```

**安装依赖：**
```bash
pip install -r requirements.txt
```

**运行应用：**
```bash
python app.py
```

## 📖 使用方法
上传 PDF 自动填充查询 → 选择分析类型与参数 → 提交并获取摘要


## 许可证
MIT License

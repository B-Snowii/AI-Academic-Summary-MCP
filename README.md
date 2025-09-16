[![Language](https://img.shields.io/badge/Language-English-blue)](README.md)
[![语言](https://img.shields.io/badge/语言-中文-red)](README.zh-CN.md)

# AI Academic Summary MCP

## 📋 Project Overview

An intelligent academic paper summarization system based on MCP (Model Context Protocol), designed for research teams to quickly generate high-quality 25-word academic summaries.

### 💡 Problems Solved
**Traditional Challenges**: Need to read 15-20 papers weekly, complete all summaries within 3-4 days, modify based on colleague feedback, ensure summaries meet team academic standards.

**AI Solution**: AI system generates drafts quickly, MCP workflow ensures efficiency, JSON configuration automatically learns team standards, continuous quality improvement based on feedback.

## 🎯 Key Outcomes
- **Time cost**: 40+ hrs/week → ≤2 hrs/week (**95% efficiency improvement**)
- **Quality assurance**: Method/topic accuracy and consistent style
- **Reusability**: MCP pipeline + historical edit data
- **Team collaboration**: Standardized output for easy colleague feedback and modification

## 🏗️ System Architecture
```
PDF Upload → Online Search → Multi-Model Reasoning → CoT Analysis → Constraint Checks → Style Rewrite → Self-Scoring → Final Summary
     ↓              ↓                ↓                ↓              ↓                ↓              ↓              ↓
Auto Parse → Real-time Info → AI + Human Collab → 4-Step Logic → Quality Control → Academic Tone → Quality Score → 25-word Output
```

### 🎯 Core Features
- **Intelligent Summary Generation**: Multi-model collaboration to generate 25-word summaries meeting academic standards
- **Real-time Online Search**: Integrated online search functionality to supplement papers with real-time academic information
- **4-Step Reasoning Framework**: Chain of Thought reasoning to ensure summary quality and accuracy
- **Team Collaboration Optimization**: Support for team feedback learning and continuous quality improvement

**Models**: Deepseek‑R1, Llama‑3.1  
**Related repo**: https://github.com/B-Snowii/Research-Paper-Summary-Collection

## 📊 Competitive Analysis

| Feature | Our System | ChatPaper | AI Summarizer |
|---------|------------|-----------|---------------|
| **Online Search** | ✅ Real-time | ❌ Static | ❌ None |
| **Multi-Model** | ✅ AI+Human | ❌ Single | ❌ Single |
| **CoT Reasoning** | ✅ 4-step | ❌ Basic | ❌ None |
| **Parameter Control** | ✅ Advanced | ❌ Fixed | ❌ None |

## 📝 Sample Summaries

Example 1: This paper examines how belief vs taste drivers shape early-stage ESG collaboration, via randomized experiments with founders and VCs; methods: randomized assignment and survey-based measures.

Example 2: This paper uses a calibrated life-cycle model to value reductions in health risks and quantify insurance, financial, and fiscal impacts; methods: structural modeling with parameter calibration.

Example 3: This paper exploits staggered adoption of hospital pay-transparency laws to study effects on patient satisfaction; methods: panel data analysis with staggered policy timing and fixed effects.

## 🧠 CoT Reasoning Framework

Our system employs a structured 4-step Chain of Thought process:

### Step 1: Topic & Method Identification
- Extract core research topic and methodology
- Identify key technical approaches and innovations

### Step 2: Content Analysis & Online Enhancement
- Analyze paper content with online search integration
- Supplement with real-time academic information

### Step 3: Constraint Validation
- Verify 25-word length requirement
- Ensure academic tone and terminology accuracy
- Validate method/topic coherence

### Step 4: Style Optimization & Self-Scoring
- Apply academic writing standards
- Self-evaluate summary quality
- Generate final polished output

## 🚀 Local Setup

**Clone the repository:**
```bash
git clone https://github.com/B-Snowii/AI-Academic-Summary-MCP.git
cd AI-Academic-Summary-MCP
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the application:**
```bash
python app.py
```

## 📖 Usage
Upload a PDF to auto-fill the query → Choose analysis type and parameters → Submit to get paper summary


## License
MIT License

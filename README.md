# 🐦 Colibri · 纯 C 轻量大模型推理引擎

<p align="center">
  <img src="https://github.com/KOO-Kunner/colibri/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  <img src="https://img.shields.io/badge/Language-C11-blue.svg" alt="C11">
  <img src="https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg" alt="Zero Dependencies">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20WSL-lightgrey.svg" alt="Platform">
</p>

> **Colibri** (蜂鸟) 是一个纯 C11 实现的轻量大语言模型 (LLM) 推理引擎。它**零第三方依赖**（仅需 libc 和 pthread），专为低硬件要求、可嵌入系统和极致性能而设计。

## ✨ 核心特性

- 🚀 **纯 C 实现**：不依赖 BLAS、CUDA 或 Python 运行时，仅需一个 C11 编译器和 `-lm -lpthread`。
- 📦 **极低门槛**：整个静态库编译后仅约 200KB，可在 512MB 内存的单核 ARM 设备上运行。
- ⚡ **极致量化**：内置 `F32`、`Q8_0` 和 `Q4_0` 量化内核，4-bit 量化可将 7B 模型的显存占用压缩至约 4GB。
- 🧠 **现代架构**：完整实现 Transformer 架构，支持 RoPE 旋转位置编码、SwiGLU 激活函数、KV Cache 缓存与 GQA（分组查询注意力）。
- 🐍 **多语言绑定**：原生提供 C API，并附带开箱即用的 Python `ctypes` 绑定。
- 📖 **代码即文档**：约 5000 行高度可读的代码，每一个执行路径都可以在半天内读完。

## 🏗️ 架构总览

Colibri 采用经典的四层分层架构，依赖方向严格自上而下：

```mermaid
graph TD
    A[应用层 CLI / Python API] --> B[运行时层 Runtime]
    B --> C[模型层 Model]
    C --> D[算子层 Ops]
    D --> E[内核层 Core / Backend]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#fbf,stroke:#333,stroke-width:2px
    style E fill:#fbb,stroke:#333,stroke-width:2px
内核层：内存分配（Arena）、哈希表、多线程池、SIMD CPU 探测、量化内核。

算子层：矩阵乘（支持量化）、RMSNorm、Softmax、SiLU、RoPE。

模型层：解析 .clb 格式、加载权重、BPE 近似分词器。

运行时层：KV Cache 管理、前向图执行、采样器（Top-K/Top-P/Greedy）、流式生成引擎。

🚀 快速开始
1. 编译 C 引擎
bash
git clone https://github.com/KOO-Kunner/colibri.git
cd colibri
make -j
# 编译产物: build/colibri, build/libcolibri.a, build/libcolibri.so
2. 生成一个玩具模型并运行（无需真实权重）
bash
# 生成一个随机的极小模型用于测试
python3 tools/make_tiny_model.py --output ./models/tiny_random.clb

# 运行推理
./build/colibri --model ./models/tiny_random.clb --prompt "Hello, world!" --n-predict 16
3. 运行 Python 绑定
bash
cd bindings/python
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

export COLIBRI_LIB=~/colibri/build/libcolibri.so
export COLIBRI_MODEL=~/colibri/models/tiny_random.clb
python3 examples/basic.py
📂 文档导航
架构设计

模型格式规范

C API 参考

贡献指南

📄 开源协议
本项目采用 MIT License 开源。你可以自由地进行商业使用、修改和闭源分发。

Built with ❤️ by The Colibri Authors.

# lang-chain-demo
学习如何使用LangChain

# 环境搭建
[使用miniconda搭建虚拟环境](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation:manual-shell-initialization)

## conda常用命令
```shell

conda create -n testenv python=3.10 # 创建一个 pythone 3.10的虚拟环境，名字为 testenv`
conda env list # 显示所有 conda 环境列表（就能看到刚刚创建的 testenv）
conda activate testenv # 激活 testenv 环境
python --version # 显示 testenv 环境中的 python 版本
conda deactivate # 退出当前 conda 环境
conda create -n testenv python # 安装最新版 python
conda search python # 查询可以创建的 python 版本
conda env remove -n testenv # 删除刚创建的 testenv 环境

```

依赖安装：常用依赖可参考requirements.txt

# 代码目录介绍
* part1: lang-chain的基础抽象示例








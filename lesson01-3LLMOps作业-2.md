# lesson01-3LLMOps作业-2

Date: 20250729-晚间00:50---07:35

作业：
    1、部署deepseek-ai/DeepSeek-R1-0528-Qwen3-8B，使用vLLM或SGLang进行部署。
    2、部署Open-WebUI来对接推理服务上的模型，完成对话。

## 1. 训练一个VLLM大模型

### 1.1创建模型实例： AutoDL: (https://www.autodl.com/home)  

充值租用1个实例，ssh暴露服务可对外接入，在界面里也可用Jupterlab直接本地访问。

创建新实例，选内蒙地区有3090显卡又便宜，今天刚上手仅为显示选择低配。

![image-20250729031734979](assets/image-20250729031734979.png)

![image-20250727165411535](assets/image-20250727165411535.png)

大模型大多数基于PyTorch, 但这里为演示手工安装PyTorch，使用conda基础镜像，按需可扩充数据盘。

![image-20250729012439702](assets/image-20250729012439702.png)

### 1.2创建python虚拟环境

#### 1.2.1使用Jupterlab

​     Jupterlab界面如下：启动页右侧加号

![image-20250727165756346](assets/image-20250727165756346.png)

Jupterlab菜单，设置，高级设置编辑器，终端，主题:dark改light, 字体大小13改15.  Ctrl+ 放大字体

<font color=red>注意各种数据不要放系统盘，否则很快打满30G。</font>

![image-20250727170258489](assets/image-20250727170258489.png)

#### 1.2.2 构建虚拟python环境
a.安装conda,  指定路径到数据盘挂载点，python 3.11环境
autodl数据盘挂载点/root/autodl-tmp/
autodl数据autodl-pub映射到根下/autodal-pub/data
已经安装miniconda3，sources.list是yum源

![image-20250729013221503](assets/image-20250729013221503.png)

b.国内源不动

![image-20250729013655646](assets/image-20250729013655646.png)

c.更新源列表
	apt update
d.安装常用软件和工具
	apt-get -y install vim iproute2 net-tools git-lfs curl wget

e.镜像站点加速安装conda config --show-sources, 添加中科大源，不加也可

```shell
root@autodl-container-4cd948a17a-3daaa04f:~# conda config --show-sources
==> /root/.condarc <==
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - defaults
show_channel_urls: True
root@autodl-container-4cd948a17a-3daaa04f:~# conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
root@autodl-container-4cd948a17a-3daaa04f:~# conda config --set show_channel_urls yes
root@autodl-container-4cd948a17a-3daaa04f:~# conda config --set always_yes yes
root@autodl-container-4cd948a17a-3daaa04f:~# conda config --show-sources
==> /root/.condarc <==
channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - defaults
show_channel_urls: True
always_yes: True
```

#### 1.2.3 终端1创建python虚拟环境

```shell
root@autodl-container-4cd948a17a-3daaa04f:~# conda create -p /root/autodl-tmp/myenv python=3.11
==> WARNING: A newer version of conda exists. <==
  current version: 22.11.1
  latest version: 25.5.1
root@autodl-container-4cd948a17a-3daaa04f:~# ls /root/autodl-tmp/
myenv
#myenv文件已创建
root@autodl-container-4cd948a17a-3daaa04f:~# ls /root/autodl-tmp/myenv/                                      bin  compiler_compat  conda-meta  include  lib  man  share  ssl  x86_64-conda-linux-gnu  x86_64-conda_cos7-linux-gnu 
#初始化conda环境
root@autodl-container-4cd948a17a-3daaa04f:~# conda init
==> For changes to take effect, close and re-open your current shell. <==
#重加载环境  source .bashrc或exec bash
root@autodl-container-4cd948a17a-3daaa04f:~# . .bashrc
#激活conda环境
(base) root@autodl-container-4cd948a17a-3daaa04f:~# conda activate /root/autodl-tmp/myenv
(/root/autodl-tmp/myenv) root@autodl-container-4cd948a17a-3daaa04f:~# 
```

#### 1.2.4 终端2安装模块modelscope vllm, pip install modelscope vllm

```shell
(base) root@autodl-container-4cd948a17a-3daaa04f:~# pip install modelscope vllm
Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.12.14 aiosignal-1.4.0 annotated-types-0.7.0 astor-0.8.1 async-timeout-5.0.1 blake3-1.0.5 cbor2-5.6.5 click-8.2.1.....
```

下载模型，例如modelscope: (https://www.modelscope.cn/home)

模型库：DeepSeek-R1-0528-Qwen3-8B  最简版 Deepseek学到知识通过蒸馏迁移到Qwen上，只保留8B参数。

地址：[DeepSeek-R1-0528-Qwen3-8B · 模型库](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/files)

16GB的大文件下载需要30-45分钟

![image-20250729022841226](assets/image-20250729022841226.png)

#### 1.2.5终端3：git lfs 大文件系统方式下载再git clone

另起终端3

```shell
(base) root@autodl-container-4cd948a17a-3daaa04f:~# which git-lfs
/usr/bin/git-lfs
(base) root@autodl-container-4cd948a17a-3daaa04f:~# cd /root/autodl-tmp/ && ls
myenv
(base) root@autodl-container-4cd948a17a-3daaa04f:~/autodl-tmp# mkdir models && cd models
(base) root@autodl-container-4cd948a17a-3daaa04f:~/autodl-tmp/models# git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B.git
Encountered 2 file(s) that may not have been copied correctly on Windows:
        model-00002-of-000002.safetensors
        model-00001-of-000002.safetensors
See: `git lfs help smudge` for more details.
(base) root@autodl-container-4cd948a17a-3daaa04f:~/autodl-tmp/models# git lfs help smudge
Known bugs
----------
On Windows, Git does not handle files in the working tree larger than 4 gigabytes.
解决方法
方法一：增加git http缓冲区，524288000Bytes=500MB
(base) root@autodl-container-4cd948a17a-3daaa04f:~/autodl-tmp/models# git config --global http.postBuffer 524288000
(base) root@autodl-container-f11943a0f2-aef7d8f5:~/autodl-tmp/models# ls -lh DeepSeek-R1-0528-Qwen3-8B/
total 16G
-rw-r--r-- 1 root root 1.1K Jul 29 03:55 LICENSE
-rw-r--r-- 1 root root  15K Jul 29 03:55 README.md
-rw-r--r-- 1 root root  859 Jul 29 03:55 config.json
-rw-r--r-- 1 root root   48 Jul 29 03:55 configuration.json
drwxr-xr-x 2 root root   35 Jul 29 03:55 figures
-rw-r--r-- 1 root root 8.1G Jul 29 04:17 model-00001-of-000002.safetensors
-rw-r--r-- 1 root root 7.3G Jul 29 04:16 model-00002-of-000002.safetensors
-rw-r--r-- 1 root root  33K Jul 29 03:55 model.safetensors.index.json
-rw-r--r-- 1 root root 6.8M Jul 29 03:55 tokenizer.json
-rw-r--r-- 1 root root 3.9K Jul 29 03:55 tokenizer_config.json
```

#### 1.2.6 终端4，检查文件下载进度

```shell
(base) root@autodl-container-4cd948a17a-3daaa04f:~# cd autodl-tmp/models/ && ls
DeepSeek-R1-0528-Qwen3-8B
(base) root@autodl-container-4cd948a17a-3daaa04f:~/autodl-tmp/models# cd DeepSeek-R1-0528-Qwen3-8B/ && ls
LICENSE  README.md  config.json  configuration.json  figures  model.safetensors.index.json  tokenizer.json  tokenizer_config.json
(base) root@autodl-container-f11943a0f2-aef7d8f5:~/autodl-tmp/models/DeepSeek-R1-0528-Qwen3-8B# for i in {1..20}; do echo -n "$i    ";du -sh;sleep 20;done
```

<font color=red>这个文件不能被ollama支持，要基于vllm方式启动</font>

```shell
安装步骤：
1.下载模型方式
    apt install git-lfs
    git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B.git
    modelscope download --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B.git

2.vllm启动命令:  VLLM_USE_MODELSCOPE=true vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B.git --tensor-parallel-size 1 --max-model-len 32768  并不支持加密验证机制例如api-key
  2.1.如已下载，无需VLLM_USE_MODELSCOPE=true，直接本地路径启动vllm serve ./DeepSeek-R1-0528-Qwen3-8B.git --tensor-parallel-size 1 --max-model-len 32768 或可加--served-model-name DeepSeek-R1-0528-Qwen3-8B不会用前面路径名做模型名，
  --tensor-parallel-size 1 几块并行GPU, --max-model-len 32768 模型最大支持Tokens数量
  2.2.没有VLLM_USE_MODELSCOPE=true 环境变量，默认HuggingFace下载，并缓存到用户家目录下隐藏目录，例如~/.cache/modelscope/hub,需注意磁盘空间是否充足。我们需要更改默认缓存目录：export HF_HOME=/root/autodl-tmp/hf_cache。
  2.3.有VLLM_USE_MODELSCOPE=true  环境变量，不从默认HuggingFace下载，而是从ModelScope下载，并缓存到用户家目录下隐藏目录，例如~/.cache/modelscope/hub，需注意磁盘空间是否充足。我们需要更改默认缓存目录：export MODELSCOPE_CACHE=/root/autodl-tmp/ms_cache。
  2.4.从HuggingFace下载需要科学上网，从huggingface 实例页面帮助文档https://autodl.com/docs 学术资源加速，在你终端中执行source /etc/network_turbo, 将上述模型缓存目录,加速方式写入系统级配置文件，~/.bashrc, ~/.bash_peofile，再source让每个终端都生效。
  2.5.vllm serve 启动vllm推理引擎 默认监听8080/tcp 
  2.6.指定模型的方法
    2.6.1 ModelID：deepseek-ai/DeepSeek-R1-0528-Qwen3-8B  默认先在本地缓存目录检查手否已下载模型，否则从HuggingFace下载
    2.6.2 ModelPath： ./DeepSeek-R1-0528-Qwen3-8B  从本地路径加载启动
  2.7.部署Open-WebUI开源对接OpenAI兼容接口的多用户前端界面，默认监听8000端口
    2.7.1 为避免与llm冲突，在终端4创建一个新的虚拟环境：conda create -p /root/autodl-tmp/open-webui python=3.11
    2.7.2 启动：open-webui serve 
  2.8.现在部署的大模型无论DS或其他，他们最大起源长度32k,32768,64k--128k, DS R1是128k, 为减少对大模型显存压力减少显存压力，限制它的最大词源数量，这里用默认。

3.还可使用SGLANG启动模型，SGLANG是另一款推理引擎，后端还是使用vllm，比vllm做了更多优化，有些环境下比vllm性能更好。
SGLANG启动命令：SGLANG_USE_MODELSCOPE=true python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --trust-remote-code --tp 1
  3.1.没有SGLANG_USE_MODELSCOPE=true  环境变量，从HuggingFace下载，并缓存到用户家目录下隐藏目录，例如~/.cache/modelscope/hub,需注意磁盘空间是否充足。我们需要更改默认缓存目录：export HF_HOME=/root/autodl-tmp/hf_cache。
  3.2.有SGLANG_USE_MODELSCOPE=true  环境变量，从默认ModelScope下载，并缓存到用户家目录下隐藏目录，例如~/.cache/modelscope/hub，需注意磁盘空间是否充足。我们需要更改默认缓存目录：export MODELSCOPE_CACHE=/root/autodl-tmp/ms_cache。
  3.3.sglang.launch_server 启动sglang推理引擎, 默认监听30000/tcp 
  3.4.--tp 1即--tensor-parallel-size 1 几块并行GPU，

4.使用SGLang来推理：还支持api-key认证机制
  SGLANG_USE_MODELSCOPE=true python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B  --trust-remote-code --tp 1 \
     --api-key magedu.com --served-model-name DeepSeek-R1-1.5B
```

#### 1.2.7终端5，创建一个新的虚拟环境安装open-webui

```shell
#网络加速
(base) root@autodl-container-f11943a0f2-aef7d8f5:~# source /etc/network_turbo
#创建open-webui虚拟环境
(base) root@autodl-container-f11943a0f2-aef7d8f5:~# conda create -p /root/autodl-tmp/open-webui python=3.11
#激活open-webui虚拟环境
(base) root@autodl-container-f11943a0f2-aef7d8f5:~# conda activate /root/autodl-tmp/open-webui
#安装open-webui
(/root/autodl-tmp/open-webui) root@autodl-container-f11943a0f2-aef7d8f5:~# pip install open-webui
#启动
(/root/autodl-tmp/open-webui) root@autodl-container-f11943a0f2-aef7d8f5:~# open-webui serve
INFO:     Started server process [7776]
INFO:     Waiting for application startup.
```

#### 1.2.8 终端2，vllm部署完成,  安装vllm modelscope组件

```shell
(base) root@autodl-container-f11943a0f2-aef7d8f5:~# pip show vllm
(base) root@autodl-container-f11943a0f2-aef7d8f5:~# pip show modelscope
```

#### 1.2.9 终端6，监控显卡控制台 nvidia-smi

![image-20250729052308290](assets/image-20250729052308290.png)

1MB模型加载到显存使用大小，这里几乎没有加载    24576MB=24GB显存大小  -l 1每隔1秒监控刷新， Ctrl+C中断

```shell
#每隔1秒执行1次，滚屏向下
(base) root@autodl-container-f11943a0f2-aef7d8f5:~# nvidia-smi -l 1
#watch -n 1，同一窗口
(base) root@autodl-container-f11943a0f2-aef7d8f5:~# watch -n 1 'nvidia-smi'
```

![image-20250729052639161](assets/image-20250729052639161.png)

在使用后模型加载到显存使用大小增加，功率24W会增加

### 1.3 加载大模型

#### 1.3.1方法一：终端1使用Model Path直接本地路径加载启动大模型

```shell
命令格式：
vllm serve ./DeepSeek-R1-0528-Qwen3-8B --tensor-parallel-size 1 --max-model-len 32768
或可加--served-model-name DeepSeek-R1-0528-Qwen3-8B不会用前面路径名做模型名
--tensor-parallel-size 1 几块并行GPU, 
--max-model-len 32768 模型最大支持Tokens数量
```

```shell
(/root/autodl-tmp/myenv) root@autodl-container-f11943a0f2-aef7d8f5:~# ls /root/autodl-tmp
models  myenv  open-webui
(/root/autodl-tmp/myenv) root@autodl-container-f11943a0f2-aef7d8f5:~# cd /root/autodl-tmp/models/ && ls
DeepSeek-R1-0528-Qwen3-8B
(/root/autodl-tmp/myenv) root@autodl-container-f11943a0f2-aef7d8f5:~/autodl-tmp/models# vllm serve ./DeepSeek-R1-0528-Qwen3-8B --tensor-parallel-size 1 --max-model-len 32768
INFO 07-29 05:30:15 [__init__.py:235] Automatically detected platform cuda.
...
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.06s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.18s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.16s/it]
...
INFO 07-29 05:36:58 [api_server.py:1818] Starting vLLM API server 0 on http://0.0.0.0:8000
...
INFO:     Started server process [9751]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
#Automatically detected platform cuda #cuda就是在GPU上直接编程利用运行程序框架。
#1个shards分片, open-webui监听本地vllm API server 8000端口
```

终端6监控，加载完成后大约需要22GB，FP16或INT8环境，功率28W

![image-20250729053938745](assets/image-20250729053938745.png)

vllm默认端口8080，vllm api server端口8000，如不成功需要科学上网下载组件或等待

```shell
(base) root@autodl-container-f11943a0f2-aef7d8f5:~# ss -ntl | grep -i -e ":8080" -e ":8000"
LISTEN 0      2048         0.0.0.0:8080       0.0.0.0:*          
LISTEN 0      2048         0.0.0.0:8000       0.0.0.0:*    
```

但是Jupyter并不对外暴露端口，我们无法直接访问,  复制登录命令与密码，自定义到本地笔记本

![image-20250729054403639](assets/image-20250729054403639.png)

![image-20250729054419489](assets/image-20250729054419489.png)



#### 1.3.2 通过ssh隧道将远程主机上端口转发到本地linux主机

```shell
建立ssh tunnel将远端主机端口转发到本地linux主机上: 远程主机ssh端口46457
    ssh -CNg -L 6006:127.0.0.1:6006 root@connect.nmb2.seetacloud.com -p 46457
修改端口为8080：
ssh -CNg -L 8080:127.0.0.1:8080 root@connect.nmb2.seetacloud.com -p 46457
-C: Compressed表示在传输数据时使用压缩。
-N: No excute表示不执行远程命令，通常情况下，SSH会登录远程服务器并执行指定的命令，但在这里，我们只是建立了一个隧道，不需要执行任何远程命令
-g: 允许远程主机连接到本地转发端口，这在一些特定的场景下是必要的。
-L 8080:127.0.0.1:8080: 这是一个本地端口转发的参数local_socket，将本地的8080端口转发到远程服务器的127.0.0.1地址的8080端口上。
-p 46457: 这是远程SSH服务器的端口号
```

本地主机终端1：本地主机ubuntu是win10 vmstation中NAT网络虚机192.168.110.60, 并未占用8080

```shell
mike@mike-virtual-machine:~/Desktop$ apt update
mike@mike-virtual-machine:~/Desktop$ apt install vim iproute2 net-tools curl wget
mike@mike-virtual-machine:~/Desktop$ ss -ntl | grep -i -e ":8080" -e ":8000"
```

本地主机终端2：创建ssh隧道，密码不会显示

```shell
mike@mike-virtual-machine:~/Desktop$ ssh -CNg -L 8080:127.0.0.1:8080 root@connect.nmb2.seetacloud.com -p 46457
```

![image-20250729055229130](assets/image-20250729055229130.png)

本地主机终端1：已建立ssh隧道

![image-20250729055425105](assets/image-20250729055425105.png)

 #### 1.3.3打开注册open-webui界面 用户名密码admin

![image-20250728175233194](assets/image-20250728175233194.png)

![image-20250728175324770](assets/image-20250728175324770.png)

#### 1.3.4 排错

默认连接本地ollama模型，后台报错，连不上ollama 11434和443端口，忽略。

![image-20250728175643382](assets/image-20250728175643382.png)

我们要将webui连接到远端部署的vllm模型，webui左侧三横菜单，选择底下admin, 管理员面板，管理员设置，外部连接，**关闭**本地没有ollama API接口,  删除本地openai 443连接。

![image-20250729000741085](assets/image-20250729000741085.png)

![image-20250728214746901](assets/image-20250728214746901.png) 

左滑关闭ollama API, 删除本地openai 443连接

![image-20250728180154030](assets/image-20250728180154030.png)

#### 1.3.5 添加OpenAI API新连接，密钥随意，vllm serve启动不支持密钥

​      一定要是<u>127.0.0.1:8000</u>，即本地主机8000--->本地主机8080--->SSL隧道--->远程vllm主机8080(vllm server)---> 远程vllm主机8000(vllm apiserver), 这里是坑。

![image-20250728222722041](assets/image-20250728222722041.png)

模型加载到了, 编辑可修改模型名

![image-20250728223437546](assets/image-20250728223437546.png)

新对话提问：神经网络基础有哪些？

![image-20250729060150965](assets/image-20250729060150965.png)

![image-20250729060226289](assets/image-20250729060226289.png)

将来大模型日志可导入Elasticsearch查询

**余额不够，作业做到这里**，但后续是DeepSeek-R1-Distill-Qwen-1.5B实验，确定能跑通。

#### 1.4.1 方法二：使用ModelID加载大模型

前端open-webui不动，Ctrl+C终止当前模型vllm运行。

#### 1.4.2 终端1, 加载环境变量

```shell
vim /root/.bashrc
文末添加
export HF_HOME=/root/autodl-tmp/hf_cache
export MODELSCOPE_CACHE=/root/autodl-tmp/ms_cache
source /etc/network_turbo
加载配置
exec bash
```

重新加载环境变量，每个终端都执行exec bash, 检查环境变量env | grep -i -e hf -e model

![image-20250728225424066](assets/image-20250728225424066.png)

#### 1.4.2 使用ModelID加载大模型

```shell
启动vllm时会先检查本地环境变量路径里是否已有缓存，没有则从modelscope下载。目录若不存在则自动创建。
VLLM_USE_MODELSCOPE=true vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --tensor-parallel-size 1 --max-model-len 32768
```

![image-20250728230403892](assets/image-20250728230403892.png)

![image-20250728231032076](assets/image-20250728231032076.png)

vllm模型启动OK。

在终端2验证，自动创建目录ms_cache并下载大模型，这是默认下载到的环境变量定义的默认缓存目录，上一版是手动下载到指定目录，也可在下载时指定缓存目录 --local-dir  ./

![image-20250728230845732](assets/image-20250728230845732.png)

#### 1.4.3 方法三：使用SGLang启动模型

1.创建SGlang虚拟环境，每个环境隔离不干扰
2.创建conda create -p /root/autodl-tmp/sglang python=3.11
3.激活conda  activate /root/autodl-tmp/sglang

![image-20250728231305466](assets/image-20250728231305466.png)

![image-20250728231554954](assets/image-20250728231554954.png)

4.SGlang对特定环境有版本需求，所以要隔离环境 pip install sglang[all]

![image-20250728231713736](assets/image-20250728231713736.png)

模型名称已变，对应models目录下路径名。

![image-20250728231942263](assets/image-20250728231942263.png)

选择这个新模型

![image-20250728232108242](assets/image-20250728232108242.png)

![image-20250728232219309](assets/image-20250728232219309.png)

SGlang激活完成。

![image-20250728232250897](assets/image-20250728232250897.png)

Ctrl+C 停止vllm服务，

![image-20250728234322630](assets/image-20250728234322630.png)

systemcache为Linux系统缓存目录
modelpath为方法一使用modelpath将大模型下载安装到指定目录，oldcache
modelID为方法二使用加载新环境变量后自动将大模型下载安装默认目录, newcache

5.启动SGLang, 

SGLANG_USE_MODELSCOPE=true python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --trust-remote-code --tp 1 \
     --api-key magedu.com --served-model-name DeepSeek-R1-1.5B

![image-20250728235556826](assets/image-20250728235556826.png)

/root/autodl-tmp/ms_cache/已存在，直接从缓存目录加载，不用重新下载。

如没有重载~/.bashrc配置文件，没有找到指定缓存目录，默认还会去加载当前用户家目录下的缓存目录~/.cache/modelscope，而不是autodl-tmp/ms_cache目录, 退出重新加载bashrc

加载完成

![image-20250729000420679](assets/image-20250729000420679.png)

终端6尽可能多地使用显存空间加速推理过程。

![image-20250729000518632](assets/image-20250729000518632.png)

改SGLang端口30000，密钥magedu.com真的有用了。

![image-20250729000902014](assets/image-20250729000902014.png)

![image-20250729000943018](assets/image-20250729000943018.png)

--serverd-model-name指定的模型名称

![image-20250729001028439](assets/image-20250729001028439.png)



![image-20250729001313180](assets/image-20250729001313180.png)

</think>包裹的是他推理过程，之下是回答。

![image-20250729001348911](assets/image-20250729001348911.png)

SGLang获得请求并提供服务。
生产环境nohup .... & 就可部署单机单卡模型，正常日志和错误日志重定向到日志。单机多卡 --tp 4就会自动切分模型分散到4个卡上运行。
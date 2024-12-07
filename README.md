# Towards Learning to Reason: Comparing LLMs with Neuro-Symbolic on Arithmetic Relations in Abstract Reasoning

### Michael Hersche, Giacomo Camposampiero, Roger Wattenhofer, Abu Sebastian and Abbas Rahimi [[Paper]]()

<div align="center">
  <img src='figs/main_figure.png' width="90%"/>
</div>

This repository contains the generation of the I-RAVEN-X dataset as well as the LLM experiments. Please refer to the [ARLC codebase](https://github.com/IBM/abductive-rule-learner-with-context-awareness) for the NeSy experiments on I-RAVEN and I-RAVEN-X.

## Build the Environment 🛠️

#### Hardware
You will need a machine with a CUDA-enabled GPU and the Nvidia SDK installed to compile the CUDA kernels. We tested our methods on a compute node with two NVIDA Tesla A100 GPUs with CUDA Version 12.1.

#### Installing Dependencies

The `micromamba` software is required for running the code. You can create a new micromamba environment using

```bash
micromamba create --name rpmllm python=3.10
micromamba activate rpmllm
```

To install PyTorch 2.3 and CUDA, use
```bash
micromamba install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```


Finally, install the requirements of this repo
```bash
pip install -r requirements.txt
pre-commit install
```

We suggest to format the code of the entire repository to improve its readability.
To do so, please install and run `black`
```bash
pip install black
black raven-with-large-language-models/
```


#### I-RAVEN and I-RAVEN-X Dataset
You can find the instructions to download and pre-process the data in the `src/data` folder.


## Run our Experiments 🔬
You can replicate the main experiments shown in the paper with the following scripts. For the GPT-4 experiments, you have to add your OpenAI API keys to `configs/models/gpt/1_gpt4_iraven.yml`.

```bash
# GPT-4 on I-RAVEN
python main.py --data-cfg configs/datasets/iraven.yml --model-cfg configs/models/gpt/1_gpt4_iraven.yml

# LLama-3 on I-RAVEN
python main.py --data-cfg configs/datasets/iraven.yml --model-cfg configs/models/llama/1_llama_iraven.yml

# GPT-4 on I-RAVEN-X
python main.py --data-cfg configs/datasets/iravenx.yml --model-cfg configs/models/gpt/2_gpt4_iravenx.yml

# LLama-3 on I-RAVEN-X
python main.py --data-cfg configs/datasets/iravenx.yml --model-cfg configs/models/llama/2_llama_iravenx.yml
```

## Citation 📚
If you use the work released here for your research, please consider citing our paper:
```
@article{hersche2024_rpm_llm_nesy,
  title={Towards Learning Abductive Reasoning using VSA Distributed Representations},
  author={Hersche, Michael and Camposampiero, Giacomo and Wattenhofer, Roger and Sebastian, Abu and Rahimi, Abbas},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```


## License 🔏
Please refer to the LICENSE file for the licensing of our code. Our implementation relies on [In-Context Analgoical Reasoning with Pre-Trained Language Models](https://github.com/hxiaoyang/lm-raven) distributed under the MIT license.

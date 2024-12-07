#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
from transformers import AutoTokenizer, pipeline


class hf_pred:
    """
    Huggingface wrapper for prediction
    """

    def __init__(
        self, name: str, nreturn: int, temperature: float, sampling: bool, **kwargs
    ):
        self._nreturn = nreturn
        self._sampling = sampling
        self._temperature = temperature
        self._max_new_tokens = 15
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = pipeline(
            "text-generation",
            model=name,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def forward(self, prompt: str, query: str):
        """
        Interaction with model
        """
        ret = []
        query = prompt + "\n" + query
        len_in = len(query)
        out = self._model(
            query,
            do_sample=self._sampling,
            temperature=self._temperature,
            num_return_sequences=self._nreturn,
            max_new_tokens=self._max_new_tokens,
        )
        for i in range(self._nreturn):
            ret.append(out[i]["generated_text"][len_in:])
        return {"prompt": query, "out": ret}


class hf_instruct_pred(hf_pred):
    """
    Huggingface wrapper instruct models
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, prompt: str, query: str):
        """
        Interaction with model
        """
        ret = []
        query = prompt + "\n " + query
        messages = [{"role": "user", "content": query}]
        len(query)
        out = self._model(
            messages,
            do_sample=self._sampling,
            temperature=self._temperature,
            num_return_sequences=self._nreturn,
            max_new_tokens=self._max_new_tokens,
        )

        for i in range(self._nreturn):
            ret.append(out[0]["generated_text"][1]["content"])
        return {"prompt": query, "out": ret}

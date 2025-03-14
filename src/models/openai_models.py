#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import time

from openai import OpenAI


class gpt_pred:
    def __init__(
        self,
        name: str,
        nreturn: int,
        temperature: float,
        sampling: bool,
        api_key: str,
        or_key: str,
        **kwargs
    ):
        self._name = name
        self._nreturn = nreturn
        self._temperature = temperature
        self._max_new_tokens = 15
        self._client = OpenAI(api_key=api_key, organization=or_key)

    def forward(self, prompt, query):
        """
        Interaction with model
        """
        ret = []
        response = self._client.chat.completions.create(
            model=self._name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=self._temperature,
            n=self._nreturn,
            max_tokens=self._max_new_tokens,
        )

        # extract all answers
        for i in range(self._nreturn):
            ret.append(response.choices[i].message.content)

        return {"system": prompt, "prompt": query, "out": ret}


class o1_pred:
    def __init__(
        self,
        name: str,
        nreturn: int,
        max_tokens: int,
        max_completion_tokens: int,
        reasoning_effort: str,
        api_key: str,
        or_key: str,
        **kwargs
    ):
        self._name = name
        self._nreturn = nreturn
        self._max_tokens = max_tokens
        self._max_completion_tokens = max_completion_tokens
        self._client = OpenAI(api_key=api_key, organization=or_key)
        self._reasoning_effort = reasoning_effort

    def forward(self, prompt, query):
        """
        Interaction with model
        """
        ret = []
        query = prompt + "\n" + query

        t_start = time.time()
        response = self._client.chat.completions.create(
            model=self._name,
            messages=[
                {"role": "user", "content": query},
            ],
            max_completion_tokens=self._max_completion_tokens,
            n=self._nreturn,
            reasoning_effort=self._reasoning_effort,
        )
        t_end = time.time()
        mytime = t_end - t_start

        # extract all answers
        for i in range(self._nreturn):
            ret.append(response.choices[i].message.content)

        # import pdb; pdb.set_trace()
        return {
            "system": "",
            "prompt": query,
            "out": ret,
            "usage": response.usage.to_dict(),
            "mytime": mytime,
        }

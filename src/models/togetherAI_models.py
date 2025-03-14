#
# Copyright 2025- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import time

from openai import OpenAI

# This is a DeepSeek R1 API using togetherAI service.


class r1_pred:
    def __init__(
        self,
        name: str,
        nreturn: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        api_key: str,
        or_key: str,
        base_url: str,
        **kwargs
    ):
        self._name = name
        self._nreturn = nreturn
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def forward(self, prompt, query):
        """
        Interaction with the R1 model
        """
        ret = []
        query = prompt + "\n" + query

        t_start = time.time()

        response = self._client.chat.completions.create(
            model=self._name,
            messages=[
                {"role": "user", "content": query},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            n=self._nreturn,
        )

        t_end = time.time()
        mytime = t_end - t_start

        # extract all answers
        for i in range(self._nreturn):
            ret.append(response.choices[i].message.content)
        usage = response.usage.to_dict()

        return {
            "system": "",
            "prompt": query,
            "out": ret,
            "usage": usage,
            "mytime": mytime,
        }

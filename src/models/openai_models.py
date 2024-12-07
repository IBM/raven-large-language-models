#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

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

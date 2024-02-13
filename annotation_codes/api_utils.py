import openai
import subprocess
import json
import ast
from typing import Any, List, Union
from tqdm import tqdm
import tiktoken
from time import sleep
import asyncio


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    return output.decode("utf-8")


def num_tokens_from_string(inputs: Union[str, list], encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string or a list of string."""
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list):
        raise ValueError(f"string must be a string or a list of strings, got {type(inputs)}")
    num_tokens = 0
    encoding = tiktoken.get_encoding(encoding_name)
    for xstring in inputs:
        num_tokens += len(encoding.encode(xstring))
    return num_tokens


def calculate_openai_api_cost(usage, model_name):
    """
    This function is used to calculate the cost of a request.
    :param usage:
    :param model_name:
    :return:
    """
    mapping = {
        "gpt-3.5-turbo-1106": (0.0010, 0.0020),
        "gpt-3.5-turbo-instruct": (0.0015, 0.0020),
        "gpt-4": (0.03, 0.06),
        "gpt-4-32k": (0.06, 0.12),
        "gpt-4-1106-preview": (0.01, 0.03),
        "text-davinci-003": (0.0010, 0.0020),
    }
    intokens = usage.prompt_tokens
    outtokens = usage.completion_tokens

    assert model_name in mapping.keys()
    return mapping[model_name][0] * intokens / 1000 + mapping[model_name][1] * outtokens / 1000


class OpenaiEngine():
    """
    This class is a simple wrapper for OpenAI API.
    """

    def __init__(self, api_key, model, way="vanilla"):
        self.api_key = api_key
        # openai.api_base = "http://34.195.96.131/v1"
        self.engine = openai.Engine(api_key=api_key)
        self.model = model
        self.way = way

    def generate(self, system_message_part, user_message_part, temperature=0.0, max_tokens=2048, top_p=1.0,
                 frequency_penalty=0, presence_penalty=0, request_timeout=120):
        if system_message_part is None:
            messages = [
                {"role": "system",
                 "content": system_message_part},
                {"role": "user",
                 "content": user_message_part}
            ]
        else:
            messages = [
                {"role": "user",
                 "content": user_message_part}
            ]
        dict_params = {"model": self.model,
                       "messages": messages,
                       "temperature": temperature,
                       "max_tokens": max_tokens,
                       "top_p": top_p,
                       "request_timeout": request_timeout,
                       }

        if self.way == "vanilla":
            completion = openai.ChatCompletion.create(
                **dict_params
            )
            text_output = completion.choices[0].message.content
            cost = calculate_openai_api_cost(completion.usage, self.model)
            # elegant_show(completion)
            # raise ValueError
            return text_output, cost
        elif self.way == "curl":
            # run a curl command
            output = run_command(
                f"""curl -s http://34.195.96.131/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" -d '{json.dumps(dict_params)}'""")
            # print(output)

            completion = json.loads(output)
            return completion["choices"][0]["message"]["content"], calculate_openai_api_cost(completion["usage"],
                                                                                             self.model)
        else:
            raise


class OpenAIChat():
    """
    This class is a more complex wrapper for OpenAI API, support async batch generation.
    """

    def __init__(self, api_key=None, model='gpt-3.5-turbo-1106',
                 temperature=0.0, max_tokens=2048, top_p=1.0,
                 frequency_penalty=0, presence_penalty=0, request_timeout=120,
                 type="text", seed=42, return_logprobs=False):
        if model == 'gpt-3.5-turbo-instruct':
            self.max_length = 4096
        elif model == 'gpt-3.5-turbo-1106':
            self.max_length = 16384
        elif model == 'gpt-4':
            self.max_length = 8192
        elif model == 'gpt-4-32k':
            self.max_length = 32768
        elif model == 'gpt-4-1106-preview':
            self.max_length = 128000
        elif model == 'text-davinci-003':
            self.max_length = 4096
        elif model == 'davinci-002':
            self.max_length = 4096
        else:
            raise ValueError('not supported model!')
        self.config = {'model_name': model, 'max_tokens': max_tokens,
                       'temperature': temperature, 'top_p': top_p,
                       'request_timeout': request_timeout, "frequency_penalty": frequency_penalty,
                       "presence_penalty": presence_penalty, "type": type, "seed": seed}
        self.return_logprobs = return_logprobs
        openai.api_key = api_key
        # openai.api_base = "http://openai.plms.ai/v1"

    def _boolean_fix(self, output):
        return output.replace("true", "True").replace("false", "False")

    def _type_check(self, output, expected_type):
        try:
            output_eval = ast.literal_eval(output)
            if not isinstance(output_eval, expected_type):
                return None
            return output_eval
        except:
            return None

    async def dispatch_openai_requests(
            self,
            messages_list,
            enable_tqdm
    ):
        """Dispatches requests to OpenAI API asynchronously.

        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        Returns:
            List of responses from OpenAI API.
        """

        async def _request_with_retry(id, messages, retry=3):
            for _ in range(retry):
                try:
                    actual_max_tokens = min(self.config["max_tokens"], self.max_length - 20 - sum(
                        [num_tokens_from_string(m['content']) for m in messages]))
                    if actual_max_tokens < self.config["max_tokens"]:
                        if actual_max_tokens > 0:
                            print(
                                f'Warning: max_tokens is too large, reduce to {actual_max_tokens} due to model max length limit ({self.max_length})!')
                        else:
                            print(f'Input is longer than model max length ({self.max_length}), aborted!')
                            return id, None
                    response = await openai.ChatCompletion.acreate(
                        model=self.config['model_name'],
                        messages=messages,
                        max_tokens=actual_max_tokens,
                        temperature=self.config['temperature'],
                        top_p=self.config['top_p'],
                        request_timeout=self.config['request_timeout'],
                        seed=self.config['seed'],
                        response_format={"type": self.config["type"]},
                        logprobs=self.return_logprobs,
                        top_logprobs=5 if self.return_logprobs else None,
                    )
                    return id, response
                except openai.error.RateLimitError:
                    print('Rate limit error, waiting for 40 second...')
                    await asyncio.sleep(40)
                except openai.error.APIError:
                    print('API error, waiting for 1 second...')
                    await asyncio.sleep(1)
                except openai.error.Timeout:
                    print('Timeout error, waiting for 1 second...')
                    await asyncio.sleep(1)
                except openai.error.ServiceUnavailableError:
                    print('Service unavailable error, waiting for 3 second...')
                    await asyncio.sleep(3)
            return id, None

        async def _dispatch_with_progress():
            async_responses = [
                _request_with_retry(index, messages)
                for index, messages in enumerate(messages_list)
            ]
            if enable_tqdm:
                pbar = tqdm(total=len(async_responses))
            tasks = asyncio.as_completed(async_responses)

            responses = []

            for task in tasks:
                index, response = await task
                if enable_tqdm:
                    pbar.update(1)
                responses.append((index, response))

            if enable_tqdm:
                pbar.close()

            responses.sort(key=lambda x: x[0])  # 根据索引排序结果

            return [response for _, response in responses]

        return await _dispatch_with_progress()

    async def async_run(self, messages_list, enable_tqdm):
        retry = 1
        responses = [None for _ in range(len(messages_list))]
        messages_list_cur_index = [i for i in range(len(messages_list))]

        while retry > 0 and len(messages_list_cur_index) > 0:
            # print(f'{retry} retry left...')
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]

            predictions = await self.dispatch_openai_requests(
                messages_list=messages_list_cur,
                enable_tqdm=enable_tqdm
            )

            # print(predictions[0])
            # raise ValueError

            preds = [
                {
                    "output": prediction['choices'][0]['message']['content'],
                    "cost": calculate_openai_api_cost(prediction['usage'], self.config["model_name"]),
                    "finish_reason": prediction['choices'][0]['finish_reason'],
                    "logprobs": prediction['choices'][0]['logprobs']
                } if prediction is not None else {"output": "Failed!", "cost": 0.0, "finish_reason": "fail"}
                for prediction in predictions
            ]

            finised_index = []
            for i, pred in enumerate(preds):
                if pred is not None:
                    responses[messages_list_cur_index[i]] = pred
                    finised_index.append(messages_list_cur_index[i])

            messages_list_cur_index = [i for i in messages_list_cur_index if i not in finised_index]

            retry -= 1

        return responses

    def generate_batch(self, msgs, direct_msg_list=False, enable_tqdm=True):
        """
        :param msgs: be like [{"sysmsg":"xx","usermsg":"yy"},...]
        :return:
        """
        if not direct_msg_list:
            msg_list = []
            for msg in msgs:
                sysmsg = msg.get("sysmsg", None)
                usermsg = msg.get("usermsg", None)
                assert usermsg is not None
                if sysmsg is None:
                    msg_list.append([{"role": "user", "content": usermsg}])
                else:
                    msg_list.append(
                        [{"role": "system", "content": msg["sysmsg"]}, {"role": "user", "content": msg["usermsg"]}])
        else:
            msg_list = msgs
        predictions = asyncio.run(self.async_run(
            messages_list=msg_list,
            enable_tqdm=enable_tqdm
        ))
        # each prediction is a tuple (response, cost)
        return predictions

    def generate_single(self, msg):
        """
        this is just a wrapper for generate_batch when only one msg is given
        :param msg: be like {"sysmsg":"xx","usermsg":"yy"}
        :return:
        """
        sysmsg = msg.get("sysmsg", None)
        usermsg = msg.get("usermsg", None)
        assert usermsg is not None
        if sysmsg is None:
            msg_list = [[{"role": "user", "content": usermsg}]]
        else:
            msg_list = [[{"role": "system", "content": msg["sysmsg"]}, {"role": "user", "content": msg["usermsg"]}]]
        predictions = asyncio.run(self.async_run(
            messages_list=msg_list,
            enable_tqdm=False
        ))
        return predictions[0]


def wrapper_p_rpair(wrapper_type, query, response_1, response_2):
    sysprompt = None
    if wrapper_type == "naive":
        strs = """You will need to analyze two responses (Response A and Response B) from AI assistants to a user's query. The query and the responses are as follows:

[Query Start]
{prompt}
[Query End]

[Response A Start]
{response_1}
[Response A End]

[Response B Start]
{response_2}
[Response B End]

Between Response A and Response B, which response is better in addressing the user's query? The better response is Response"""
    else:
        raise NotImplementedError

    return sysprompt, strs.format(prompt=query, response_1=response_1, response_2=response_2)

import os
import json
import pandas as pd


def print_colored_text(text, color="yellow", end=None):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }

    color_code = colors.get(color.lower(), colors["reset"])
    print(f"{color_code}{text}{colors['reset']}", end=end)


def read_parquet(parquet_file_path):
    data = pd.read_parquet(parquet_file_path)
    data = data.to_dict("records")
    return data


def write_jsonl(data, jsonl_file_path, mode="w"):
    # data is a list, each of the item is json-serilizable
    assert isinstance(data, list)
    if not os.path.exists(os.path.dirname(jsonl_file_path)):
        os.makedirs(os.path.dirname(jsonl_file_path))
    with open(jsonl_file_path, mode) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def write_json(data, json_file_path):
    if not os.path.exists(os.path.dirname(json_file_path)):
        os.makedirs(os.path.dirname(json_file_path))
    with open(json_file_path, "w") as f:
        json.dump(data, f)


def read_jsonl(jsonl_file_path):
    s = []
    if not os.path.exists(jsonl_file_path):
        print_colored_text("File not exists: " + jsonl_file_path, "red")
        return s
    with open(jsonl_file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        linex = line.strip()
        if linex == "":
            continue
        s.append(json.loads(linex))
    return s


def read_json(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return data


def read_all(file_path):
    if file_path.endswith(".jsonl"):
        return read_jsonl(file_path)
    elif file_path.endswith(".json"):
        return read_json(file_path)
    elif file_path.endswith(".parquet"):
        return read_parquet(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip() != ""]
    else:
        raise ValueError(f"Unrecognized file type: {file_path}")


# Function to convert JSON to Markdown formatted string with bold keys
def json_to_markdown_bold_keys(json_obj, depth=0):
    markdown_str = ""
    indent = "\t "
    for key, value in json_obj.items():
        if isinstance(value, dict):
            markdown_str += f"**{key}** :\n\n{indent * (depth)}- {json_to_markdown_bold_keys(value, depth + 1)}\n\n"
        elif isinstance(value, list):
            if len(value) > 0:
                markdown_str += (
                    f"**{key}** :\n\n "
                    + f"\n\n{indent * (depth)}- "
                    + f"\n\n{indent * (depth)}- ".join(
                        [
                            (
                                json_to_markdown_bold_keys(item, depth + 1)
                                if isinstance(item, dict)
                                else f"{indent * (depth + 1)}{item}"
                            )
                            for item in value
                        ]
                    )
                    + "\n\n"
                )
            else:
                markdown_str += f"**{key}** : None\n\n"
        else:
            if depth == 0:
                markdown_str += f"**{key}** : {value}\n\n"
            else:
                markdown_str += f"{indent * (depth)}- **{key}** : {value}\n\n"
    return markdown_str


def custom_md_with_color(text, color):
    return f"""
<div style="background-color:#{color};padding:10px;border-radius:5px;">
    <p style="color:black;font-size:16px;">📑\n\n{text}</p>
</div>"""

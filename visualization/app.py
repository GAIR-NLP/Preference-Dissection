import streamlit as st
import os
from utils import read_all, json_to_markdown_bold_keys, custom_md_with_color
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import pandas as pd
import json

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from matplotlib import pyplot as plt
import shap
from functools import partial


import base64

numpyro.set_host_device_count(4)

feature_name_to_id = {
    "harmlessness": 0,
    "grammar, spelling, punctuation, and code-switching": 1,
    "friendly": 2,
    "polite": 3,
    "interactive": 4,
    "authoritative tone": 5,
    "funny and humorous": 6,
    "metaphors, personification, similes, hyperboles, irony, parallelism": 7,
    "complex word usage and sentence structure": 8,
    "use of direct and explicit supporting materials": 9,
    "well formatted": 10,
    "admit limitations or mistakes": 11,
    "persuade user": 12,
    "step by step solution": 13,
    "use of informal expressions": 14,
    "non-repetitive": 15,
    "clear and understandable": 16,
    "relevance without considering inaccuracy": 17,
    "innovative and novel": 18,
    "information richness without considering inaccuracy": 19,
    "no minor errors": 20,
    "no moderate errors": 21,
    "no severe errors": 22,
    "clarify user intent": 23,
    "showing empathetic": 24,
    "satisfying explicit constraints": 25,
    "supporting explicit subjective stances": 26,
    "correcting explicit mistakes or biases": 27,
    "length": 28,
}

feature_name_to_id_short = {
    "harmless": 0,
    "grammarly correct": 1,
    "friendly": 2,
    "polite": 3,
    "interactive": 4,
    "authoritative": 5,
    "funny": 6,
    "use rhetorical devices": 7,
    "complex word & sentence": 8,
    "use supporting materials": 9,
    "well formatted": 10,
    "admit limits": 11,
    "persuasive": 12,
    "step-by-step": 13,
    "use informal expressions": 14,
    "non-repetitive": 15,
    "clear": 16,
    "relevant": 17,
    "novel": 18,
    "contain rich info": 19,
    "no minor errors": 20,
    "no moderate errors": 21,
    "no severe errors": 22,
    "clarify intent": 23,
    "show empathetic": 24,
    "satisfy constraints": 25,
    "support stances": 26,
    "correct mistakes": 27,
    "lengthy": 28,
}

small_mapping_for_query_specific_cases = {
    "w_constraints": "Contain Explicit Constraints",
    "w_stances": "Show Explicit Subjective Stances",
    "w_mistakes": "Contain Mistakes or Bias",
    "intent_unclear": "Unclear User Intent",
    "express_feeling": "Express Feelings of Emotions",
}

preset_model_order_in_paper_w_size = {
    "yi-6b":6,
    "yi-6b-chat":6,
    "llama-2-7b":7,
    "llama-2-7b-chat":7,
    "vicuna-7b-v1.5":7,
    "tulu-2-dpo-7b":7,
    "mistral-7b":7,
    "mistral-7b-instruct-v0.1":7,
    "mistral-7b-instruct-v0.2":7,
    "zephyr-7b-alpha":7,
    "zephyr-7b-beta":7,
    "qwen-7b":7,
    "qwen-7b-chat":7,
    "llama-2-13b":13,
    "llama-2-13b-chat"  :13,
    "wizardLM-13b-v1.2":13,
    "vicuna-13b-v1.5":13,
    "tulu-2-dpo-13b":13,
    "qwen-14b":14,
    "qwen-14b-chat":14,
    "yi-34b":34,
    "yi-34b-chat"   :34,
    "mistral-8x7b":56,
    "mistral-8x7b-instruct-v0.1":56,
    "llama-2-70b":70,
    "llama-2-70b-chat":70,
    "wizardLM-70b-v1.0":70,
    "tulu-2-dpo-70b":70,
    "qwen-72b":72,
    "qwen-72b-chat" :72,
    "gpt-3.5-turbo-1106":500,
    "gpt-4-1106-preview":2000,
    "human":10000
}

feature_id_to_name_short = {v: k for k, v in feature_name_to_id_short.items()}

feature_names_short = list(feature_name_to_id_short.keys())

all_models_fitted_params = {}

def formal_group_name(part):
    if part[0].isupper():
        part = f"[Scenario] {part}"
    else:
        part = f"[Query-Specific Cases] {small_mapping_for_query_specific_cases[part]}"
    return part

# add the fitted parameters in papers
for fn in os.listdir(f"./data/fitted_paras_comparison"):
    part = fn[len("model_"): fn.find("_fitted_paras")]
    part = formal_group_name(part)
    if part not in all_models_fitted_params:
        all_models_fitted_params[part] = {}
    dd = read_all(f"./data/fitted_paras_comparison/{fn}")
    for it in dd:
        all_models_fitted_params[part][it["model_name"]] = it["parameters"]

# add newly added models by users
community_model_size = {}
for fn in os.listdir(f"./data/fitted_paras_comparison_community"):
    ffn = f"./data/fitted_paras_comparison_community/{fn}"
    newly_added = read_all(ffn)
    size_model_name = fn[:-len(".json")]
    size,model_name = size_model_name.split("_")
    community_model_size[model_name] = float(size)
    for part in newly_added:
        assert part in all_models_fitted_params
        all_models_fitted_params[part][model_name] = newly_added[part]


modelwise_fitted_paras = {}
for group in all_models_fitted_params:
    for model in all_models_fitted_params[group]:
        if model not in modelwise_fitted_paras:
            modelwise_fitted_paras[model] = {}
        modelwise_fitted_paras[model][group] = all_models_fitted_params[group][model]


def show_one_model_prob(weights, feature_names=None):
    plt.figure(figsize=(20, 7))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    all_probabilities = []

    weights = np.asarray(weights)
    posterior_means = weights
    X_test = np.eye(weights.shape[0])

    logits = X_test @ posterior_means
    probabilities = 100 / (1 + np.exp(-logits))
    all_probabilities.extend(probabilities)

    plt.scatter(
        range(0, weights.shape[0]),
        probabilities,
        label='apple',
        s=380,
        alpha=0.65,
    )

    min_prob = min(all_probabilities)
    max_prob = max(all_probabilities)
    plt.ylim([min_prob - 3, max_prob + 3])

    # plt.xlabel('Feature Names')
    plt.ylabel("Probability of Preferred (%)")
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    if feature_names is not None:
        plt.xticks(range(0, len(feature_names)), feature_names, rotation=45, ha="right")
    else:
        plt.xticks(range(0, weights.shape[0]), ha="center")

    plt.grid(True)
    plt.axhline(y=50, color="red", linestyle="--")

    plt.subplots_adjust(bottom=0.3, right=0.85)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()


def show_all_models_prob(models, selected_models, feature_names=None):
    plt.figure(figsize=(17, 7))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    all_probabilities = []
    for model_name in selected_models:
        weights = np.asarray(models[model_name])
        posterior_means = weights
        X_test = np.eye(weights.shape[0])

        logits = X_test @ posterior_means
        probabilities = 100 / (1 + np.exp(-logits))
        all_probabilities.extend(probabilities)

        plt.scatter(
            range(0, weights.shape[0]),
            probabilities,
            label=model_name,
            s=380,
            alpha=0.65,
        )

    min_prob = min(all_probabilities)
    max_prob = max(all_probabilities)
    plt.ylim([min_prob - 3, max_prob + 3])

    # plt.xlabel('Feature Names')
    plt.ylabel("Probability of Preferred (%)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    if feature_names is not None:
        plt.xticks(range(0, len(feature_names)), feature_names, rotation=45, ha="right")
    else:
        plt.xticks(range(0, weights.shape[0]), ha="center")

    plt.grid(True)
    plt.axhline(y=50, color="red", linestyle="--")

    plt.subplots_adjust(bottom=0.3, right=0.85)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()


def process_query_info(x):
    s = []
    if x["clear intent"] != "Yes":
        s.append("[Query-Specific Cases] Unclear User Intent")
    if x["explicitly express feelings"] == "Yes":
        s.append("[Query-Specific Cases] Express Feelings of Emotions")
    if len(x["explicit constraints"]) > 0:
        s.append("[Query-Specific Cases] Contain Explicit Constraints")
    if len(x["explicit subjective stances"]) > 0:
        s.append("[Query-Specific Cases] Show Explicit Subjective Stances")
    if len(x["explicit mistakes or biases"]) > 0:
        s.append("[Query-Specific Cases] Contain Mistakes or Bias")
    return s


def get_feature(item, remove_length=False, way="comparison"):
    # way be "comparison" or "diff" or "norm_diff"
    feature = [0] * len(feature_name_to_id)
    comparison = item["comparison"]
    for k, v in comparison.items():
        if k == "accuracy":
            for xx in ["Severe", "Moderate", "Minor"]:
                feature[feature_name_to_id[f"no {xx.lower()} errors"]] = v[way][xx]
        elif k == "repetitive":
            feature[feature_name_to_id["non-repetitive"]] = -v[way]
        else:
            feature[feature_name_to_id[k]] = v[way]
    if remove_length:
        feature = feature[:-1]
    return feature


class BayesianLogisticRegression:
    def __init__(self, alpha):
        self.alpha = alpha

    def predict(self, X):
        probs = self.return_prob(X)
        predictions = np.round(probs)
        return predictions

    def return_prob(self, X):
        logits = np.dot(X, self.alpha)
        # return probabilities
        return np.exp(logits) / (1 + np.exp(logits))


def bayesian_logistic_regression(X, y, scale=0.01):
    # Priors for the regression coefficients
    alpha = numpyro.sample('alpha', dist.Laplace(loc=jnp.zeros(X.shape[1]), scale=scale))

    # Calculate the linear predictor (the logits) using JAX NumPy
    logits = jnp.dot(X, alpha)

    # Likelihood of the observations given the logistic model
    with numpyro.plate('data', X.shape[0]):
        numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)


def fit_bayes_logistic_regression(X, y, scale=0.1, ):
    # repeat X and y on the first axis to get more samples

    bxx = partial(bayesian_logistic_regression, scale=scale)

    kernel = NUTS(bxx)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=2000, num_chains=4, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), X, y)

    # Get the posterior samples
    posterior_samples = mcmc.get_samples()

    # Compute the mean of the posterior for each alpha_i
    alpha_mean = np.mean(posterior_samples['alpha'], axis=0).tolist()

    return BayesianLogisticRegression(alpha_mean), alpha_mean


def get_similarity(dict1, dict2, type="pearson", select_part="Overall"):
    assert dict1.keys() == dict2.keys(), "Dicts must have the same keys"
    if select_part == "Overall":
        all_sim = 0.0
        count = 0.0
        for key in dict1.keys():
            if key.startswith("[Query-Specific Cases]"): continue
            sim = get_similarity_local(dict1[key], dict2[key], type)
            all_sim += sim
            count += 1
        return all_sim / count
    else:
        return get_similarity_local(dict1[select_part], dict2[select_part], type)


def get_similarity_local(list1, list2, type="pearson"):
    """
    Calculate the similarity between two lists of numbers based on the specified type.

    :param list1: a dict, each field is a list of floats
    :param list2: a dict, each field is a list of floats
    :param type: which kind of 'similarity' is calculated
    :return: the calculated similarity
    """
    assert len(list1) == len(list2), "Lists must be of the same length"

    if type == "pearson":
        # Pearson correlation
        similarity, _ = pearsonr(list1, list2)
    elif type == "spearman":
        # Spearman correlation
        similarity, _ = spearmanr(list1, list2)
    elif type == "normed_l1":
        # Normalized negative L1 norm (Manhattan distance)
        similarity = -np.sum(np.abs(np.array(list1) - np.array(list2))) / len(list1)
    elif type == "normed_l2":
        # Normalized negative L2 norm (Euclidean distance)
        similarity = -np.sqrt(np.sum((np.array(list1) - np.array(list2)) ** 2)) / len(
            list1
        )
    else:
        raise NotImplementedError("The specified similarity type is not implemented")

    return similarity


@st.cache_resource
def calculate_similarity_matrix(
        modelwise_fitted_paras, selected_models, similarity_type, selected_part
):
    # Initialize a matrix to store similarities
    if similarity_type in ["spearman", "pearson"]:
        similarity_matrix = np.ones((len(selected_models), len(selected_models)))
    else:
        similarity_matrix = np.zeros((len(selected_models), len(selected_models)))

    # Calculate similarities
    for i, model1 in enumerate(selected_models):
        for j, model2 in enumerate(selected_models):
            if i < j:  # Calculate only for upper triangular matrix
                sim = get_similarity(
                    modelwise_fitted_paras[model1],
                    modelwise_fitted_paras[model2],
                    similarity_type,
                    selected_part,
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric matrix
    return similarity_matrix


def format_matrix(matrix):
    formatted_matrix = np.array(matrix, dtype=str)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            formatted_matrix[i, j] = f"{matrix[i, j]:.2f}".lstrip("0")
    return formatted_matrix


def become_formal(name):
    name = (
        name.replace("6b", "6B")
        .replace("7b", "7B")
        .replace("13b", "13B")
        .replace("14b", "14B")
        .replace("34b", "34B")
        .replace("70b", "70B")
        .replace("72b", "72B")
    )
    name = (
        name.replace("llama", "LLaMA")
        .replace("yi", "Yi")
        .replace("mistral", "Mistral")
        .replace("qwen", "Qwen")
        .replace("tulu", "Tulu")
        .replace("vicuna", "Vicuna")
        .replace("wizardLM", "WizardLM")
        .replace("zephyr", "Zephyr")
    )
    name = name.replace("chat", "Chat")
    name = name.replace("gpt-3.5-turbo-1106", "GPT-3.5-Turbo").replace(
        "gpt-4-1106-preview", "GPT-4-Turbo"
    )
    name = (
        name.replace("instruct", "Inst").replace("dpo", "DPO").replace("human", "Human")
    )
    return name


def display_markdown_with_scroll(text, height=200):
    """
    Display the given Markdown text in a scrollable area using <pre> tag.

    Args:
    text (str): The Markdown text to be displayed.
    height (int): Height of the scrollable area in pixels.
    """
    # 使用 <pre> 标签来包裹 Markdown 内容，并添加 CSS 样式创建可滚动的区域
    markdown_container = f"""
    <pre style="
        overflow-y: scroll;
        height: {height}px;
        border: 1px solid #ccc;
        padding: 10px;
        margin-bottom: 20px;
        background-color: #f5f5f5;
    ">
    {text}
    </pre>
    """

    st.markdown(markdown_container, unsafe_allow_html=True)


@st.cache_resource
def compute_one_model_fitted_params(filename, num_fold, query_aware_idxs, resolved_data):
    st.write('---------------')
    one_model_fitted_params = {}
    data = json.load(filename)
    uploaded_labels = [1 if x == "A" else 0 for x in data]

    ccount=0

    for part in list(query_aware_idxs.keys()):
        if part == "all": continue
        # 使用 st.empty 创建占位符
        progress_text = st.empty()
        # if part not in ["Advice","NLP Tasks"]:continue
        progress_text.write(f"{ccount+1}/{len(list(query_aware_idxs.keys()))-1} "+formal_group_name(part))
        progress_bar = st.progress(0)
        cared_idxs = query_aware_idxs.get(part)

        features = []
        labels = []

        for idx, item in enumerate(resolved_data):
            if idx not in cared_idxs: continue
            if item['comparison']['accuracy']['comparison'] == 999: continue
            label = uploaded_labels[idx]
            feature = get_feature(item, remove_length=False, way='comparison')
            features.append(feature)
            labels.append(label)

        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels)

        if num_fold>1:
            np.random.seed(0)
            idxs = np.arange(len(features))
            np.random.shuffle(idxs)
            features = features[idxs]
            labels = labels[idxs]

            final_paras = None
            for i in range(num_fold):
                # take the i/10 as test set
                features_len = len(features)
                split_point = int(i / num_fold * features_len)
                features_train, features_test = np.concatenate(
                    [features[:split_point, :], features[split_point + int(features_len / num_fold):, :]],
                    axis=0), features[split_point:split_point + int(features_len / num_fold), :]
                labels_train, labels_test = np.concatenate(
                    [labels[:split_point], labels[split_point + int(features_len / num_fold):]], axis=0), labels[
                                                                                                          split_point:split_point + int(
                                                                                                              features_len / 10)]
                model, parameters = fit_bayes_logistic_regression(features_train, labels_train, scale=0.1)
                if final_paras is None:
                    final_paras = np.asarray(parameters)
                else:
                    final_paras += np.asarray(parameters)
                progress_bar.progress((i + 1)/num_fold)
        else:
            model, parameters = fit_bayes_logistic_regression(features, labels, scale=0.1)
            final_paras = np.asarray(parameters)
            progress_bar.progress(1)

        final_paras /= num_fold
        parameters = final_paras.tolist()
        one_model_fitted_params[formal_group_name(part)] = parameters

        # 函数处理完毕，清除进度条和文本
        progress_text.empty()
        progress_bar.empty()
        ccount+=1

    return one_model_fitted_params

def get_json_download_link(json_str, file_name, button_text):
    # 创建一个BytesIO对象
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{file_name}">{button_text}</a>'
    return href


if __name__ == "__main__":
    st.title("Visualization of Preference Dissection")

    INTRO = """    
This space is used to show visualization results for human and LLM preferences analyzed in the following paper:


[***Dissecting Human and LLM Preferences***](https://arxiv.org/abs/2402.11296)

by [Junlong Li](https://lockon-n.github.io/), [Fan Zhou](https://koalazf99.github.io/), [Shichao Sun](https://shichaosun.github.io/), [Yikai Zhang](https://arist12.github.io/ykzhang/), [Hai Zhao](https://bcmi.sjtu.edu.cn/home/zhaohai/) and [Pengfei Liu](http://www.pfliu.com/)

------------

Specifically, we include:

1. **Complete Preference Dissection in Paper**: shows how the difference of properties in a pair of responses can influence different LLMs'(human included) preference. <br>
2. **Preference Similarity Matrix**: shows the preference similarity among different judges. <br>
3. **Sample-level SHAP Analysis**: applies shapley value to show how the difference of properties in a pair of responses affect the final preference. <br>
4. **Add a New Model for Preference Dissection**: update the preference labels from a new LLM and visualize the results

This analysis is based on:

> The data we collected here: https://huggingface.co/datasets/GAIR/preference-dissection

> The code we released here: https://github.com/GAIR-NLP/Preference-Dissection
"""
    message = custom_md_with_color(INTRO, "DBEFEB")

    st.markdown(message, unsafe_allow_html=True)

    st.write("## :red[⬇] Click the Box and Select a Section :red[⬇]")

    section = st.selectbox(
        "",
        [
            "Complete Preference Dissection in Paper",
            "Preference Similarity Matrix",
            "Sample-level SHAP Analysis",
            'Add a New Model for Preference Dissection'
        ],
    )
    st.markdown("---")

    if section == "Complete Preference Dissection in Paper":
        st.header("Complete Preference Dissection in Paper")
        st.markdown("")
        selected_part = st.selectbox(
            "**Scenario/Query-Specific Cases**", list(all_models_fitted_params.keys())
        )

        models = all_models_fitted_params[selected_part]

        model_names = list(models.keys())
        selected_models = st.multiselect(
            "**Select LLMs (Human) to display**",
            model_names,
            default=["human", "gpt-4-1106-preview"],
        )

        st.text(
            "The value for each property indicates that, when response A satisfies only this\nproperty better than response B and all else equal, the probability of response\nA being preferred.")

        if len(selected_models) > 0:
            show_all_models_prob(models, selected_models, feature_names_short)
        else:
            st.write("Please select at least one model to display.")
    elif section == "Preference Similarity Matrix":
        st.header("Preference Similarity Matrix")

        # Initialize session state for similarity matrix

        # convert `groupwise_fitted_paras` to `modelwise_fitted_paras`

        models = list(modelwise_fitted_paras.keys())
        # Option to choose between preset models or selecting models
        option = st.radio(
            "**Choose your models setting**",
            ("Preset Models in Paper",
             "All Models (including newly added)"
             ,"Select Models Manually")
        )

        if option == "Preset Models in Paper":
            selected_models = preset_model_order_in_paper_w_size
        elif option == "All Models (including newly added)":
            all_model_size = {**preset_model_order_in_paper_w_size, **community_model_size}
            # sort the keys
            ## 1. by size
            ## 2. if size is the same, by name.lower()
            # return the sorted keys
            print(all_model_size)
            selected_models = sorted(all_model_size.keys(), key=lambda x: (all_model_size[x], x.lower()))
        else:
            selected_models = st.multiselect(
                "**Select Models**", models, default=models[:5]
            )

        # Input for threshold value
        st.text(
            "The similarity bewteen two judges is the average pearson correlation coefficient of\nthe fitted Bayesian logistic regression models' weights across all scenarios.")

        selected_part = st.selectbox(
            "**Overall or Scenario/Query-Specific Cases**", ["Overall"] + list(all_models_fitted_params.keys())
        )

        st.text(
            "\"Overall\" is the average similarity across all scenarios, \nwhile \"Scenario/Query-Specific Cases\" is the similarity within \nthe selected scenario/query-specific cases.")

        if len(selected_models) >= 2:
            # Call the cached function
            similarity_matrix = calculate_similarity_matrix(
                modelwise_fitted_paras, selected_models, "pearson", selected_part
            )
            # Store the matrix in session state
            # Slider to adjust figure size
            fig_size = (
                25
                if option == "Use Preset Models"
                else int(33 * len(selected_models) / 25)
            )

            plt.figure(figsize=(fig_size * 1.1, fig_size))
            ax = sns.heatmap(
                similarity_matrix,
                annot=True,
                annot_kws={"size": 18},  # Change annotation font size
                xticklabels=[become_formal(x) for x in selected_models],
                yticklabels=[become_formal(x) for x in selected_models],
            )

            # Add this line to get the colorbar object
            cbar = ax.collections[0].colorbar

            # Here, specify the font size for the colorbar
            for label in cbar.ax.get_yticklabels():
                # label.set_fontsize(20)  # Set the font size (change '10' as needed)
                label.set_fontname(
                    "Times New Roman"
                )  # Set the font name (change as needed)

            plt.xticks(rotation=45, fontname="Times New Roman", ha="right")
            plt.yticks(rotation=0, fontname="Times New Roman")

            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.warning("Please select at least two models.")
    elif section == "Sample-level SHAP Analysis":
        st.header("Sample-level SHAP Analysis")
        resolved_data_file = "./data/chatbot_arena_no-tie_group_balanced_resolved.jsonl"
        source_data_file = "./data/chatbot_arena_shuffled_no-tie_group_balanced.jsonl"
        reference_data_file = (
            "./data/chatbot_arena_shuffled_no-tie_gpt4_ref_group_balanced.jsonl"
        )

        # Load and prepare data
        resolved_data, source_data, reference_data = (
            read_all(resolved_data_file),
            read_all(source_data_file),
            read_all(reference_data_file),
        )
        ok_idxs = [
            i
            for i, item in enumerate(resolved_data)
            if item["comparison"]["accuracy"]["comparison"] != 999
        ]
        resolved_data, source_data, reference_data = (
            [resolved_data[i] for i in ok_idxs],
            [source_data[i] for i in ok_idxs],
            [reference_data[i] for i in ok_idxs],
        )
        features = np.asarray(
            [
                get_feature(item, remove_length=False, way="comparison")
                for item in resolved_data
            ],
            dtype=np.float32,
        )

        # Initialize the index
        if "sample_ind" not in st.session_state:
            st.session_state.sample_ind = 0


        # Function to update the index
        def update_index(change):
            st.session_state.sample_ind += change
            st.session_state.sample_ind = max(
                0, min(st.session_state.sample_ind, len(features) - 1)
            )


        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])

        with col1:
            st.button("Prev", on_click=update_index, args=(-1,))

        with col3:
            number = st.number_input(
                "Go to sample:",
                min_value=0,
                max_value=len(features) - 1,
                value=st.session_state.sample_ind,
            )
            if number != st.session_state.sample_ind:
                st.session_state.sample_ind = number

        with col5:
            st.button("Next", on_click=update_index, args=(1,))

        # Use the updated sample index
        sample_ind = st.session_state.sample_ind

        reference, source, resolved = (
            reference_data[sample_ind],
            source_data[sample_ind],
            resolved_data[sample_ind],
        )

        groups = [f"[Scenario] {source['group']}"] + process_query_info(
            resolved["query_info"]
        )

        st.write("")
        group = st.selectbox(
            "**Scenario & Potential Query-Specific Cases:**\n\nWe set the scenario of this sample by default, but you can also select certain query-specfic groups if the query satisfy certain conditions.",
            options=groups,
        )
        model_name = st.selectbox(
            "**The Preference of which LLM (Human):**",
            options=list(all_models_fitted_params[group].keys()),
        )
        paras_spec = all_models_fitted_params[group][model_name]
        model = BayesianLogisticRegression(paras_spec)
        explainer = shap.Explainer(model=model.return_prob, masker=np.zeros((1, 29)))

        # Calculate SHAP values
        shap_values = explainer(
            features[st.session_state.sample_ind: st.session_state.sample_ind + 1, :]
        )
        shap_values.feature_names = list(feature_name_to_id_short.keys())

        # Plotting

        st.markdown(
            "> *f(x) > 0.5 means response A is preferred more, and vice versa.*"
        )
        st.markdown(
            "> *Property = 1 means response A satisfy the property better than B, and vice versa. We only show the properties that distinguish A and B.*"
        )

        # count how mant nonzero in shape_values[0].data
        nonzero = np.nonzero(shap_values[0].data)[0].shape[0]
        shap.plots.waterfall(shap_values[0], max_display=nonzero + 1, show=False)
        fig = plt.gcf()
        st.pyplot(fig)

        # st.subheader(
        #     "**Detailed information (source data and annotation) of this sample.**"
        # )

        # We pop some attributes first

        # RAW Json
        simplified_source = {
            "query": source["prompt"],
            f"response A ({source['model_a']}, {source['response_a word']} words)": source[
                "response_a"
            ],
            f"response B ({source['model_b']}, {source['response_b word']} words)": source[
                "response_b"
            ],
            "GPT-4-Turbo Reference": reference["output"],
        }
        simplified_resolved = {
            "query-specific:": resolved["query_info"],
            "Annotation": {
                k: v["meta"]
                for k, v in resolved["comparison"].items()
                if v["meta"] is not None and k != "length"
            },
        }

        # Source Data Rendering
        # st.json(simplified_source)
        st.write("#### Source Data")
        st.text_area(
            "**Query**:\n",
            f"""{source["prompt"]}\n""",
        )
        st.text_area(
            f"**response A ({source['model_a']}, {source['response_a word']} words)**:\n",
            f"""{source["response_a"]}\n""",
            height=200,
        )
        st.text_area(
            f"**response B ({source['model_b']}, {source['response_b word']} words)**:\n",
            f"""{source["response_b"]}\n""",
            height=200,
        )
        st.text_area(
            f"**GPT-4-Turbo Reference**:\n",
            f"""{reference["output"]}\n""",
            height=200,
        )

        # Resolved Data Rendering
        st.markdown("---")
        st.write("### Annotation")
        # st.json(simplified_resolved)
        st.write("#### Query Information\n")
        query_info = json_to_markdown_bold_keys(simplified_resolved["query-specific:"])
        st.markdown(custom_md_with_color(query_info, "DFEFDB"), unsafe_allow_html=True)

        specific_check_feature_fixed = [
            "length",
            "accuracy",
        ]
        specific_check_feature_dynamic = [
            "clarify user intent",
            "showing empathetic",
            "satisfying explicit constraints",
            "supporting explicit subjective stances",
            "correcting explicit mistakes or biases"
        ]
        specific_check_feature = specific_check_feature_fixed + specific_check_feature_dynamic
        normal_check_feature = {
            k: v["meta"]
            for k, v in resolved["comparison"].items()
            if v["meta"] is not None and k not in specific_check_feature
        }
        # generate table for normal check feature
        data = {"Category": [], "Response 1": [], "Response 2": []}

        for category, responses in normal_check_feature.items():
            # print(responses)
            data["Category"].append(category)
            data["Response 1"].append(responses["Response 1"])
            data["Response 2"].append(responses["Response 2"])

        df = pd.DataFrame(data)

        # Display the table in Streamlit
        st.write("#### Ratings of Basic Properties\n")
        st.table(df)

        # specific check features: 'accuracy', and 'satisfying explicit constraints'
        st.write("#### Error Detection")

        # xx
        acc1 = simplified_resolved["Annotation"]["accuracy"]["Response 1"]
        newacc1 = {"applicable to detect errors": acc1["accuracy check"],
                   "detected errors": acc1["inaccuracies"]}
        acc2 = simplified_resolved["Annotation"]["accuracy"]["Response 2"]
        newacc2 = {"applicable to detect errors": acc2["accuracy check"],
                   "detected errors": acc2["inaccuracies"]}

        # Convert the JSON to a Markdown string
        response_1 = json_to_markdown_bold_keys(newacc1)
        response_2 = json_to_markdown_bold_keys(newacc2)
        st.markdown("##### Response 1")
        st.markdown(custom_md_with_color(response_1, "DBE7EF"), unsafe_allow_html=True)
        st.text("")
        st.markdown("##### Response 2")
        st.markdown(custom_md_with_color(response_2, "DBE7EF"), unsafe_allow_html=True)

        if any(j in simplified_resolved['Annotation'] for j in specific_check_feature_dynamic):
            st.text("")
            st.markdown("#### Query-Specific Annotation")

            for j in specific_check_feature_dynamic:
                if j in simplified_resolved['Annotation']:
                    st.write(f"**{j} (ratings from 0-3 or specific labels)**")
                    st.markdown(custom_md_with_color(json_to_markdown_bold_keys(simplified_resolved['Annotation'][j]),
                                                     "E8DAEF"), unsafe_allow_html=True)
                    st.text("")
    else:
        st.header("Add a New Model for Preference Dissection")
        resolved_data = read_all("./data/chatbot_arena_no-tie_group_balanced_resolved.jsonl")
        query_aware_idxs = read_all("./data/query_aware_idxs.json")

        st.write("Upload the preference labels from a new LLM.")
        st.write("The data in ths .json file should be a list with 5240 (the same as the data size) elements, each belongs to {\"A\",\"B\"} indicating the preferred one in each pair.")
        st.write("We provide an example in ```./data/example_preference_labels.json``` in the ``Files`` of the space, which are the preference labels of human.")
        filename = st.file_uploader("", type=["json"],
                                    key="new_model_fitted_params")

        one_model_fitted_params = None

        if filename is not None:
            st.write("Uploaded successfully.")

            st.write("Please select the number of folds for fitting the models. 1 means no multi-fold averaging. (Warning! Large number of fold may cause OOM and the crush of this space.)")
            num_fold = st.selectbox("Number of Folds", [1, 2, 5, 10], index=0)

            one_model_fitted_params = compute_one_model_fitted_params(filename, num_fold, query_aware_idxs,
                                                                      resolved_data)

        if one_model_fitted_params is not None:
            json_data = json.dumps(one_model_fitted_params, indent=4)
            st.markdown(get_json_download_link(json_data, "fitted_weights.json", "Download Fitted Bayesian Logistic Models Weights"), unsafe_allow_html=True)

            st.write("The visualization is the same as the first section.")

            selected_part = st.selectbox("**Scenario/Query-Specific Cases**", list(one_model_fitted_params.keys()))
            weights = one_model_fitted_params[selected_part]
            show_one_model_prob(weights, feature_names_short)
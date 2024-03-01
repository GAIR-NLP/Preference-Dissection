import argparse
import json
from app import *


def compute_one_model_fitted_params(filename, num_fold, query_aware_idxs, resolved_data):
    one_model_fitted_params = {}

    with open(filename, "r") as f:
        data = json.load(f)

    uploaded_labels = [1 if x == "A" else 0 for x in data]

    ccount = 0

    for part in list(query_aware_idxs.keys()):
        if part == "all": continue
        # 使用 st.empty 创建占位符
        # if part not in ["Advice","NLP Tasks"]:continue
        print(f"{ccount + 1}/{len(list(query_aware_idxs.keys())) - 1} " + part)
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

        if num_fold > 1:
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
        else:
            model, parameters = fit_bayes_logistic_regression(features, labels, scale=0.1)
            final_paras = np.asarray(parameters)

        final_paras /= num_fold
        parameters = final_paras.tolist()
        one_model_fitted_params[formal_group_name(part)] = parameters

        ccount += 1

    return one_model_fitted_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Test-Model")
    parser.add_argument("--model_size", type=float, default=6.5, help="the size of the target model")
    args = parser.parse_args()

    model_name = args.model_name
    model_size = args.model_size

    filename = f"./data/added_model_inference/{model_size}_{model_name}/direct_ask_preferences.json"
    num_fold = 10  # we set 10 here, which is the default value
    query_aware_idxs = read_all("./data/query_aware_idxs.json")
    resolved_data = read_all("./data/chatbot_arena_no-tie_group_balanced_resolved.jsonl")

    fitted_params = compute_one_model_fitted_params(filename, num_fold, query_aware_idxs, resolved_data)

    with open(f"./data/fitted_paras_comparison_community/{model_size}_{model_name}.json", "w") as f:
        json.dump(fitted_params, f)

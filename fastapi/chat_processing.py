from typing import List, Dict, Union, Tuple, cast

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel


def make_predict(
        sim_model: BertModel,
        tokenizer: BertTokenizer,
        questions: List[str],
        answer: str
) -> List[float]:
    relevances = []

    for question in questions:
        query = "[CLS]" + question + "RESPONSE_TOKEN" + answer
        tokenized = tokenizer(query, max_length=512, add_special_tokens=False, return_tensors='pt')

        with torch.inference_mode():
            logits = sim_model(**tokenized).logits
            probas = sim_model.sigmoid(logits)[0].cpu().detach().numpy()
        relevance, _ = probas
        relevances.append(relevance)

    return relevances

def get_message_tree(
        messages_dataframe: pd.DataFrame,
        sim_model: BertModel,
        tokenizer: BertTokenizer,
        subgroup_id: Union[int, None] = None,
):
    graph = {}
    root_nodes = set()

    data = messages_dataframe.dropna(subset=["text"]).reset_index(drop=True).assign(text=lambda df: df["text"].str.slice(stop=500))

    false_reply = "море море океаны море море"


    for i, row in enumerate(data.itertuples()):
        msg_id = getattr(row, "id", None)
        reply_to = getattr(row, "reply_to_msg_id", None)
        text = getattr(row, "text", "")

        if msg_id is None:
            continue

        parent = int(reply_to) if reply_to == reply_to else None
        try:
            if (parent is subgroup_id) and i > 0:
                if len(text.split()) <= 4:
                    parent = int(data.iloc[i-1, "id"])
                else:
                    messages_for_rerank = data.iloc[i - 2:i]["text"].tolist()
                    messages_for_rerank.append(false_reply)
                    predict_relevance = make_predict(sim_model, tokenizer, messages_for_rerank, text)

                    max_relevance = max(predict_relevance)

                    if predict_relevance[-1] == max_relevance or predict_relevance[-2] < 0.65:
                        parent = None
                    else:
                        parent = int(data.iloc[i - 1]['id'])

        except Exception as e:
            ...

        graph[msg_id] = {"parent": parent, "children": [], "message": text}

        if parent is subgroup_id:
            root_nodes.add(msg_id)
        else:
            if parent not in graph:
                graph[parent] = {"parent": None, "children": [msg_id], "message": ''}
            else:
                graph[parent]["children"].append(msg_id)
            if graph[parent]["parent"] is None:
                root_nodes.add(parent)

    return graph, root_nodes

def bfs(node: int, message_list: List[str], graph: Dict) -> None:
    children = graph[node]['children']
    for child in children:
        message_list.append(graph[child]['message'])
        bfs(child, message_list, graph)

def create_dialog_branches(
        data: pd.DataFrame,
        sim_model: BertModel,
        tokenizer: BertTokenizer,
        subgroup_id: Union[int, None] = None
) -> List[Tuple[str, str]]:
    graph, root_nodes = get_message_tree(
        data,
        sim_model=sim_model,
        tokenizer=tokenizer,
        subgroup_id=subgroup_id
    )

    all_message_branches = []

    for root_node in root_nodes:
        message_list = [graph[root_node]['message']]
        bfs(root_node, message_list, graph)
        if len(message_list) > 1:
            all_message_branches.append((str(root_node), cast(str, '\n'.join(message_list))))

    return all_message_branches
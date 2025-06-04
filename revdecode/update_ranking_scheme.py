import json

'''
This script is used to update the ranking scheme. The original ranking scheme is like:
[[1], [2,2,2], [3,3], [4], [5]], the number here represent the rank number.
The new ranking scheme is like:
[[1], [2,2,2], [5,5], [7], [8]].
The main purpose for this change is to do detailed performance analysis based on NDCG metric.
'''


def extract_ranking(data_list):
    """
    Extracts the ranking format from the data list.

    Args:
    - data_list: List[Dict[str, Union[str, int]]].
      A list of dictionaries containing 'rank' and potentially other attributes.

    Returns:
    - List[List[int]]. The intermediate format of ranks.
    """
    rank_list = [item['rank'] for item in data_list]
    intermediate_ranking = []
    while rank_list:
        value = rank_list.pop(0)
        sublist = [value]
        while rank_list and rank_list[0] == value:
            sublist.append(rank_list.pop(0))
        intermediate_ranking.append(sublist)
    return intermediate_ranking


def transform_ranking(intermediate_ranking):
    """
    Transforms the intermediate ranking to the new ranking format.

    Args:
    - intermediate_ranking: List[List[int]]. The intermediate format of ranks.

    Returns:
    - List[List[int]]. The transformed ranks in intermediate format.
    """
    new_ranking = []
    prev_rank = 0
    for sublist in intermediate_ranking:
        new_rank = prev_rank + 1
        new_ranking.append([new_rank] * len(sublist))
        prev_rank = new_rank + len(sublist) - 1
    return new_ranking


def create_new_data_with_rank(data_list, transformed_ranking):
    """
    Creates a new list with the transformed ranking and the original data.

    Args:
    - data_list: List[Dict[str, Union[str, int]]]. The original list with unmodified ranks.
    - transformed_ranking: List[List[int]]. The ranks in transformed format.

    Returns:
    - List[Dict[str, Union[str, int]]]. New data list with transformed ranks.
    """
    new_data = []
    data_index = 0
    for sublist in transformed_ranking:
        for rank in sublist:
            updated_item = data_list[data_index].copy()
            updated_item['rank'] = rank
            new_data.append(updated_item)
            data_index += 1
    return new_data


def updating_ranking_for_one_target_function(ranking_info):
    """
    Updates the ranking for one target function.

    Args:
    - ranking_info: List[Dict[str, Union[str, int]]]. The original ranking information.
    """
    intermediate = extract_ranking(ranking_info)
    transformed = transform_ranking(intermediate)
    updated_ranking = create_new_data_with_rank(ranking_info, transformed)

    return updated_ranking

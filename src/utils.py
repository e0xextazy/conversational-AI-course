from aiogram import types
from langchain_core.messages import HumanMessage

from typing import List, Dict, Any, Tuple, Union


def create_kb():
    kb = [[types.KeyboardButton(text="/clear_history")], [types.KeyboardButton(
        text="/help"), types.KeyboardButton(text="/get_stat")]]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
    )

    return keyboard


def convert2langchain_format(messages: List[Tuple[str, str]]) -> List[Union[HumanMessage, str]]:
    res = []
    for (h_m, ai_m) in messages:
        res += [HumanMessage(h_m), ai_m]
    return res


def aggregate_stat(history: List[Tuple[str, str]]) -> Dict[str, Any]:
    stat = {}
    stat["len_history"] = len(history)
    stat["avg_len_msg"] = int(
        (sum([len(el[1].split()) for el in history]) / stat["len_history"]) * 100) / 100
    stat["avg_toxic"] = int(
        (sum([el[3] for el in history]) / stat["len_history"]) * 100) / 100

    return stat

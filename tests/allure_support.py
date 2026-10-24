# -*- coding: utf-8 -*-
"""Allure 可选依赖垫片：未安装 allure-pytest 时装饰器自动退化为 no-op，
保证 `pip install -r requirements.txt` 最小集（pytest+requests）也能跑。"""
from __future__ import annotations


def _noop_decorator(*args, **kwargs):
    # 支持两种用法：@allure.feature("x") 与被直接当 @allure.step 裸调用
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


try:
    import allure  # type: ignore
except ImportError:  # pragma: no cover
    class _AllureShim:
        def __getattr__(self, name: str):
            return _noop_decorator

    allure = _AllureShim()  # type: ignore

__all__ = ["allure"]

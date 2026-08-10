# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.abstract_parser import (
    DelegatingParser,
    Parser,
    _WrappedParser,
)
from vllm.parser.parser_manager import ParserManager

__all__ = [
    "Parser",
    "DelegatingParser",
    "ParserManager",
    "_WrappedParser",
]

_PARSERS_TO_REGISTER = {
    "minimax_m2": (  # name
        "minimax_m2_parser",  # filename
        "MiniMaxM2Parser",  # class_name
    ),
    # vllm/parser/inkling.py ships an InklingParser whose config declares
    # name="inkling", but nothing registered it, so --reasoning-parser inkling
    # raised "has not been registered". It is a unified ParserEngine handling
    # both reasoning and tool calls, so passing the same name to
    # --reasoning-parser and --tool-call-parser takes ParserManager.get_parser's
    # strategy 1 and returns it for both.
    "inkling": (
        "inkling",
        "InklingParser",
    ),
}


def register_lazy_parsers():
    for name, (file_name, class_name) in _PARSERS_TO_REGISTER.items():
        module_path = f"vllm.parser.{file_name}"
        ParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_parsers()

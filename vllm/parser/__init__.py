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
    # Inkling drives reasoning and tool calls from ONE state machine, so it has
    # to be registered here as a unified parser. Without this entry
    # ParserManager.get_parser falls through to _WrappedParser, which runs the
    # reasoning and tool engines as two independent instances and slices the
    # text between them at the reasoning boundary. Non-streaming survives that
    # (it is a single pass through one engine) but streaming does not: the
    # tool-call phase never opens and `<|content_invoke_tool_json|>{...}` is
    # emitted verbatim as assistant content. Every streaming client -- pi
    # included -- then sees a tool call as literal text and never invokes it.
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

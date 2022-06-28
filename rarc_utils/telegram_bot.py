"""Telegram_bot.py, utility methods for telegram bots.

If this file get larger, restructure it into a new package solely for Telegram helper methods
"""

from collections import OrderedDict
from typing import Dict, List

from telegram.ext import CommandHandler, Dispatcher


def toEscapeMsg(msg: str) -> str:
    """Escape all symbols that have special meaning in Markdown."""
    return (
        msg.replace("[", "\\[")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("-", "\\-")
        .replace("+", "\\+")
        .replace("=", "\\=")
        .replace(".", "\\.")
        .replace("`", "\\`")
    )
    # replace("_", "\\_")
    # .replace("*", "\\*") \


def get_handler_docstrings(dp: Dispatcher, sortAlpha=True) -> Dict[str, str]:
    """Create list of commands to show as menu.

    Install commands by talking to BotFather: /setcommands,
    select bot and paste the returning string of this method

    !!! Only uses first line of docstring, use pydocstring to enforce this as style guide.
    """
    command_handlers = [
        i for i in list(dp.handlers.values())[0] if isinstance(i, CommandHandler)
    ]
    handler_dict = {ch.command[0]: ch.callback for ch in command_handlers}

    if sortAlpha:
        handler_dict = OrderedDict(sorted(handler_dict.items()))

    docstring_dict: Dict[str, str] = {
        command: callback.__doc__.split("\n")[0]
        for command, callback in handler_dict.items()
    }

    return docstring_dict


def create_set_commands_string(dd: Dict[str, str]) -> str:
    """Parse docstring_dict to a format BotFather can understand.

    Example:
        command1 - Description
        command2 - Another description

    Usage:
        from rarc_utils.telegram_bot import create_set_commands_string, get_handler_docstrings
        # dp = updater.dispatcher
        dd = get_handler_docstrings(dp)
        print(create_set_commands_string(dd))
    """
    command_msgs: List[str] = [" - ".join(tpl) for tpl in list(dd.items())]

    return "\n".join(command_msgs)

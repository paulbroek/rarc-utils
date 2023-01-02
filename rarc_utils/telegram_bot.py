"""Telegram_bot.py, utility methods for telegram bots.

If this file gets larger, restructure it into a new package solely for Telegram helper methods
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, TypeVar

from telegram.ext import CommandHandler

# Dispatcher removed in bot v20?
# from telegram.ext import Dispatcher
Dispatcher = TypeVar("Dispatcher", bound=Any)



logger = logging.getLogger(__name__)


class MissingDocstring(Exception):
    pass


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

    missing_docstrings = [c for c, cb in handler_dict.items() if cb.__doc__ is None]
    nmissing = len(missing_docstrings)

    if nmissing > 0:
        raise MissingDocstring(
            f"docstrings missing for ({nmissing}): {missing_docstrings}"
        )

    # warn user if docstring is missing
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


def delete_messages(dp: Dispatcher, messages: Sequence[Dict[str, Any]]) -> int:
    """Delete messages, return number of succesful deletions."""
    ret = []

    for message in messages:
        try:
            res = dp.bot.delete_message(
                chat_id=message["chat_id"], message_id=message["message_id"]
            )
            ret.append(res)
        except Exception as e:
            logger.error(f"cannot delete message. {e=!r}")

    return sum(ret)


def delete_conv_msgs(dp: Dispatcher, key="conversation_messages") -> None:
    """Delete conversations messages that are still visible due to bot restart / crash."""
    for user_id, user_data in dp.persistence.user_data.items():
        messages = user_data.get(key, {})
        res = delete_messages(dp, messages.values())
        logger.info(f"deleted {res:,} messages for {user_id=}")

        # reset msgs
        user_data[key] = {}

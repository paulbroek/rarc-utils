"""Telegram_bot.py, utility methods for telegram bots.

If this file gets larger, restructure it into a new package solely for Telegram helper methods
"""

import logging
from collections import OrderedDict
from typing import (Any, Callable, Dict, List, Optional, Sequence, TypeVar,
                    Union)

from telegram.ext import Application, CommandHandler

Dispatcher = TypeVar("Dispatcher", bound=Any)


logger = logging.getLogger(__name__)


class MissingDocstring(Exception):
    """Docstring is missing for Telegram CommandHandler."""

    pass


class MessageNotFound(Exception):
    """Telegram message not found in conversation."""

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


# def as_handler_dict_old(command_handlers: List[CommandHandler]) -> Dict[str, Callable]:
#     return {ch.command[0]: ch.callback for ch in command_handlers}


def as_handler_dict_V20(command_handlers: List[CommandHandler]) -> Dict[str, Callable]:
    return {list(ch.commands)[0]: ch.callback for ch in command_handlers}


def extract_description_from_callback(callback: Callable) -> str:
    docstring: Optional[str] = callback.__doc__
    if docstring is not None:
        return str(docstring.split("\n")[0])
    return ""


def get_handler_docstrings(
    app: Union[Application, Dispatcher],
    sortAlpha: bool = True,
    asHandlerDict: Callable[
        [List[CommandHandler]], Dict[str, Callable]
    ] = as_handler_dict_V20,
) -> Dict[str, str]:
    """Create list of commands to show as menu.

    Install commands by talking to BotFather: /setcommands,
    select bot and paste the returning string of this method

    ! Should be used with pydocstring, to make sure first docstring line contains description.
    """
    command_handlers: List[CommandHandler] = [
        i for i in list(app.handlers.values())[0] if isinstance(i, CommandHandler)
    ]
    handlers_by_name: Dict[str, Callable] = asHandlerDict(command_handlers)

    if sortAlpha:
        handlers_by_name = OrderedDict(sorted(handlers_by_name.items()))

    docstrings_by_callback_name: Dict[str, Optional[str]] = {
        c: cb.__doc__ for c, cb in handlers_by_name.items()
    }
    missing_docstrings: List[str] = [
        c for c, docstring in docstrings_by_callback_name.items() if docstring is None
    ]
    # raise when docstring is missing
    if (nmissing := len(missing_docstrings)) > 0:
        raise MissingDocstring(
            f"docstrings missing for ({nmissing}): {missing_docstrings}"
        )

    descriptions_by_name: Dict[str, str] = {
        command: extract_description_from_callback(callback)
        for command, callback in handlers_by_name.items()
    }

    return descriptions_by_name


def create_set_commands_string(dd: Dict[str, str]) -> str:
    """Parse docstring_dict to a format BotFather can understand.

    Example:
        command1 - Description
        command2 - Another description

    ----------------------------------
    Usage:
        from rarc_utils.telegram_bot import create_set_commands_string, get_handler_docstrings
        # dp = updater.dispatcher
        dd = get_handler_docstrings(dp)
        print(create_set_commands_string(dd))
    """
    command_msgs: List[str] = [" - ".join(tpl) for tpl in list(dd.items())]

    return "\n".join(command_msgs)


def delete_messages(
    app: Application | Dispatcher, messages: Sequence[Dict[str, Any]]
) -> int:
    """Delete bot messages, return number of succesful deletions."""
    ret: List[bool] = []

    for message in messages:
        try:
            res: bool = app.bot.delete_message(
                chat_id=message["chat_id"], message_id=message["message_id"]
            )
            ret.append(res)
        except Exception as e:
            raise MessageNotFound from e

    return sum(ret)


def delete_conv_msgs(dp: Dispatcher, key="conversation_messages") -> None:
    """Delete conversations messages that are still visible due to bot restart / crash."""
    for user_id, user_data in dp.persistence.user_data.items():
        messages = user_data.get(key, {})
        res = delete_messages(dp, messages.values())
        logger.info(f"deleted {res:,} messages for {user_id=}")

        # reset msgs
        user_data[key] = {}

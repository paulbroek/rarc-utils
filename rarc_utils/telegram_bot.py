"""Telegram_bot.py, utility methods for telegram bots."""


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

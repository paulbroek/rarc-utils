""" telegram_bot.py 

    utility methods for telegram bots
"""


def toEscapeMsg(msg: str) -> str:
    """markdown messages cannot contain markers of style like _ * = etc."""

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

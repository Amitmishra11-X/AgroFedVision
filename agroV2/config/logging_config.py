import logging

def get_logger(name):

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)s | %(message)s"

    )

    stream = logging.StreamHandler()

    stream.setFormatter(formatter)

    logger.addHandler(stream)

    return logger
import logging


def message(l_type, msg):
    """
    Log message and print to console if classified as info type
    """
    if l_type == logging.INFO:
        logging.info(msg)
        print(msg)
    elif l_type == logging.WARNING:
        logging.warning(msg)
    elif l_type == logging.DEBUG:
        logging.debug(msg)

import logging

def get_logger(name:str='kvsched'):
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)

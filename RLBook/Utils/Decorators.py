# -*- coding: utf-8 -*-
""" RLBook.Utils.Decorators
"""
import logging
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        logging.info('Method: {} - {} seconds'.format(method.__name__,
                                                      time.time() - ts))
        return result

    return timed

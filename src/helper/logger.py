# -*- coding: utf-8 -*-

import logging
import os

from csv_logger import CsvLogger


def get_logger(name, log_dir):
    """Creates a logger object

    :param name: name of the logger file
    :type name: str
    :param log_dir: directory where logger file needs to be stored
    :type log_dir: str
    :return: logger that writes to csv file
    :rtype: csv_logger.CsvLogger
    """
    filename = os.path.join(log_dir, name.replace("/", "-"))
    level = logging.DEBUG
    header = ["date", "epoch", "type", "measure", "confusion_matrix"]

    # Creat logger with csv rotating handler
    csvlogger = CsvLogger(filename=filename, level=level, header=header)

    return csvlogger

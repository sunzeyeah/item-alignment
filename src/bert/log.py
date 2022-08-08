# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :commodity-alignment
# @File     :log
# @Date     :2022/8/7 15:09
# @Author   :mengqingyang
# @Email    :mengqingyang0102@163.com
-------------------------------------------------
"""
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


LOGGER = logging.getLogger(__name__)

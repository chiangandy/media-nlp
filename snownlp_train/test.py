# -*- coding: utf-8 -*-
import os
import re

from snownlp import SnowNLP
from snownlp import sentiment

def test_process():
    nlp_str = [ u"""沒看卻能知道被嘴，幽默，而且也太自以為，被嗆個幾句在那邊鄭重道歉，我你老師笑到差點從椅子上掉下來""",
                u"""能這樣做已經很好了，不然妳想怎樣""",
                u"""好還要更好，一定要讓你知道""",
                u"""妳的好只有我知道""",
                u"""學點東西就出來現""",
                u"""妳只有這點能耐嗎？""",
                u"""不要再當酸民了，好嗎？""",
                u"""最漂亮的你，依舊動人可愛""",
                u"""妳想我嗎，我很想你""",
                u"""妳的自信讓我發光"""
    ]
    for str in nlp_str:
        s = SnowNLP(str)
        print "score:", "%.5f" % s.sentiments, str

if __name__ == '__main__':
    test_process()
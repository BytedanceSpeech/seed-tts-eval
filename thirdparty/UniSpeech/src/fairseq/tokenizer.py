# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    if not isinstance(line, str):
        return line
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

def char_tokenizer(line):
    line = line.strip().replace(' ', '|')+'|'
    char_list = []
    char_list[:0] = line
    return char_list

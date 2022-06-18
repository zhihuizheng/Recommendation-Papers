#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import urllib

""" generate readme.md """
__author__ = 'heyneo'

# unlists = ['.DS_Store', '.git', '.idea']
dirname = os.path.abspath('./')
out_file = open('./README.md', 'w')


def gn(dir, prefix, dir_or_file):
    def match(filename, dir_or_file):
        if dir_or_file == "dir":
            return os.path.isdir(filename)
        elif dir_or_file == "file":
            return os.path.isfile(filename)
        else:
            return False

    lines = []
    for name in sorted(os.listdir(dir)):
        if name.startswith('.'): continue
        filename = os.path.join(dir, name)
        if match(filename, dir_or_file):
            lines.append(prefix + name)
    return '\n\n'.join(lines)


lines = []
for name in sorted(os.listdir(dirname)):
    if name.startswith('.'): continue
    filename = os.path.join(dirname, name)
    if os.path.isdir(filename):
        lines.append("## " + name)
        line = gn(filename, "", dir_or_file="file")
        lines.append(line)
out_file.write("\n\n".join(lines))
out_file.close()

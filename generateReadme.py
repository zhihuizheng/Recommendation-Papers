#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import urllib.parse

""" generate readme.md """

sort_reverse = False

paper_class_map = {}
paper_map = {}

out_file = open('./README.md', 'w')


github_root = "https://github.com/zhihuizheng/Recommendation-Papers/blob/main/"
all_dir = os.listdir("./")
all_dir.sort()

for one_dir in all_dir:
    if os.path.isdir(one_dir) and not one_dir.startswith('.'):
        out_file.write("\n## " + one_dir + "\n")
        if one_dir.strip() in paper_class_map:
            out_file.write(paper_class_map[one_dir.strip()] + "\n")
        all_sub_files = os.listdir(one_dir)
        all_sub_files.sort(reverse=sort_reverse)

        for one_file in all_sub_files:
            one_file_2 = os.path.join(one_dir, one_file)
            if not os.path.isdir(one_file_2) and not one_file.startswith('.'):
                out_file.write(
                    "* [" + ('.').join(one_file.split('.')[:-1]) + "](" + github_root + one_dir.strip() + "/"
                    + urllib.parse.quote(one_file.strip()) + ") <br />\n"
                )
                if one_file.strip() in paper_map:
                    out_file.write(paper_map[one_file.strip()] + "\n")

        all_sub_files.sort(reverse=sort_reverse)

        for one_file in all_sub_files:
            one_file_2 = os.path.join(one_dir, one_file)
            if os.path.isdir(one_file_2) and not one_file_2.startswith('.'):
                one_dir_second = one_file_2
                out_file.write("\n#### " + one_file + "\n")
                all_sub_files_second = os.listdir(one_dir_second)
                all_sub_files_second.sort(reverse=sort_reverse)

                for one_file_second in all_sub_files_second:
                    if not os.path.isdir(one_file_second) and not one_file_second.startswith('.'):
                        out_file.write(
                            "* [" + ('.').join(one_file_second.split('.')[:-1]) + "](" + github_root
                            + urllib.parse.quote(one_dir_second.strip()) + "/"
                            + urllib.parse.quote(one_file_second.strip()) + ") <br />\n"
                        )

out_file.close()
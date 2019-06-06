import csv
import sys
sys.path.append("../dataProcessing")
import data_processing as dp

# read in csv data
with open("./mbti_1.csv", 'r', encoding="utf-8") as f:
    itr = csv.reader(f)
    next(itr)

    d = {}
    types = []
    posts = []
    for type, post in itr:
        types.append(type)
        posts.append(post)

    d["types"] = types
    d["posts"] = posts


dp.preprocess_string(d["posts"][0])

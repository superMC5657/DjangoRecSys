# -*- coding: utf-8 -*-
# !@time: 2022/5/31 下午9:41
# !@author: superMC @email: 18758266469@163.com
# !@fileName: app.py

from flask import Flask, url_for, render_template, request, redirect, session, jsonify
import os
import sys
import io
import random
from flask import Response

from sql_utils import fetch_collection_name

basepath = os.path.dirname(__file__)

app = Flask(__name__)
app.debug = True
sys.path.append(os.path.join(basepath, 'text_classify'))
sys.path.append(os.path.join(basepath, 'text_ner'))
from text_classify.inference import inf_classify
from text_ner.inference import inf_ner
from text_similarity.inference import inf_q_sim, inf_info_sim

d_threshold = 0.5
a_threshold = 0.5
cache = None

second = ['藏品编号', '藏品名称', '藏品类别', '藏品年代']


@app.route("/main", methods=["GET", "POST"])
def simulate_test0():
    global cache
    print('in simulate_test0')
    flag = request.form.get("flag")
    text = request.form.get("text")
    print('flag=', flag, 'text=', text)

    if flag not in second:
        res = ''
        relics_list = inf_ner(text)
        cls = inf_classify(text)
        print(cls)
        print(relics_list)
        if relics_list:
            r_list = []
            for relics in relics_list:
                str_list = fetch_collection_name(relics)
                if str_list:
                    r_list += str_list
            print(r_list)
            cache = r_list
            if r_list:
                for r in r_list:
                    res += r + '\n'
                print(res)
                if len(r_list) > 1:
                    return {"status": "more", "content": res}, 200
                return {"status": "sucess", "content": res}, 200
            else:
                return {"status": "failed", "content": "不好意思，这个问题系统无法解答，接入人工客服请咨询010-9646"}, 200
        else:

            desc, d_p = inf_info_sim(text)
            ans, a_p = inf_q_sim(text)
            print(ans, a_p)
            print(desc, d_p)

            if d_p > a_p and d_p >= d_threshold:
                res = desc
                return {"status": "sucess", "content": res}, 200

            elif a_p >= a_threshold:
                res = ans
                return {"status": "sucess", "content": res}, 200
            else:
                return {"status": "failed", "content": "不好意思，这个问题系统无法解答，接入人工客服请咨询010-9646"}, 200



    else:
        restore = []
        for relics_str in cache:
            if text in relics_str:
                restore.append(relics_str)
        res = ''
        for r in restore:
            res += r + '\n'
        if len(restore) > 1:
            cache = restore
            return {"status": "more", "content": res}, 200
        return {"status": "sucess", "content": res}, 200


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host='localhost', port=8000, threaded=False)

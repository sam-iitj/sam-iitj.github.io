---
layout: post
comments: true
title:  "Deploying ML models - Part 1"
excerpt: "Deploy ML model using Flask"
date:   2016-11-04 15:40:00
mathjax: false
---

### Flask

Flask is a lightweight Python web framework to create microservices. Wanna read [more](https://code.tutsplus.com/tutorials/an-introduction-to-pythons-flask-framework--net-28822) ? I am a ML guy, and this sounds complex :-( Rather than reading lets code a simple one quickly !

#### Install Flask

```python
pip install flask
```

I used python 2.7 and Flask==0.11.1

#### Bare bones Example

Open am editor and copy paste code from my [git repo](https://github.com/anujgupta82/Musings/blob/master/flask/simple_app.py)

```python
from flask import Flask

app = Flask(__name__)

@app.route('/1')   # path to resource on server
def index_1():        # action to take
  return "Hello_world 1"

@app.route('/2')
def index_2():
  return "Hello_world 2"

if __name__ == '__main__':
  app.run(debug=True)
```

To run this:

    1. Save it as simple_app.py    
    2. Install Flask in your virtual environment [pip install Flask]     
    3. Open terminal, go to the directory where app.py is saved. Run following two commands     

```python
export FLASK_APP=simple_app.py
flask run
```

This should have the flask server up and running on http://127.0.0.1:5000

<div class="imgcap">
<img src="/assets/ml_models_1/image_1.png">
<div class="thecap">Flask Server up and running</div>
</div>



If you see the code carefully it says - we have 2 resources with relative URIs as '/1' and '/2'. Lets access them. Go to browser and type http://127.0.0.1:5000/1

This should fire the function index_1() function and give the following output on the command prompt

<div class="imgcap">
<img src="/assets/ml_models_1/image_2.png">
<div class="thecap">Output from function index_1()</div>
</div>

Like wise http://127.0.0.1:5000/2 should work. This is a simple flask application. (Oh yeah! this sounds easy, lets move on)
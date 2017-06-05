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

#### REST (###### in peace)

There are a couple of terms that are part and parcel on micro services. Lets quickly  understand something about them.

    1) API : Application Program Interface - set of routines, protocols, and tools for building software applications.     

    2) API Endpoint :It's one end of a communication channel, so often this would be represented as the URL of a server or service. In our example "http://127.0.0.1:5000/1"      
    
    3) REST :underlying architectural principle of the web. 
    Read these awesome [stackoverflow answer](https://stackoverflow.com/questions/671118/what-exactly-is-restful-programming/671132#671132) and this brilliant [post](http://web.archive.org/web/20130116005443/http://tomayko.com/writings/rest-to-my-wife) from Ryan Tomayko and this [post](https://martinfowler.com/articles/richardsonMaturityModel.html) from Martin Fowler to understand the same.     

In nutshell, you need to have - GET, POST, PUT, DELETE.

Lets add this to our [code](https://github.com/anujgupta82/Musings/blob/master/flask/RESTful_app.py). To see this in action, run the server (like previously), go to terminal and type

```python
curl -i http://localhost:5000/tasks
```

or 

```python
curl -i -X GET http://localhost:5000/tasks
```

Both the commands will give same output:

<div class="imgcap">
<img src="/assets/ml_models_1/image_3.png">
<div class="thecap">GET request</div>
</div>


Your server terminal will show "200" (success) for both the requests.

<div class="imgcap">
<img src="/assets/ml_models_1/image_4.png">
<div class="thecap">200 – success</div>
</div>

#### RESTful App

Lets add other parts of RESTful to out code. Here it is. To see this in action, run the server (like previously), go to terminal and type:

 	1) Get All tasks:    

```python
curl -i http://localhost:5000/tasks/
```         

 	2) Get a specific task:

```python
curl -i http://localhost:5000/tasks/2
```    

<div class="imgcap">
<img src="/assets/ml_models_1/image_5.png">
<div class="thecap">Get task with id=2</div>
</div>

	Since there is no task with id=4, try this:
	
```python
curl -i http://localhost:5000/tasks/4
```

<div class="imgcap">
<img src="/assets/ml_models_1/image_6.png">
<div class="thecap">Error. Task with id=4 does not exists</div>
</div>



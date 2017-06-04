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

### Install Flask

```python
s = "Python syntax highlighting"
print s
pip install flask
```

I have often heard Data scientist/ML people asking "I have a ML model thats doing well on test data, how do I deploy it in production environment ?"  Recently I have been exploring the same, so sharing some of my findings and ways to achieve the same. Here focus will be on deployment and not developing a model - hence, for simplicity, we will assume we have a pre-trainied, fine tuned [scikit model](http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html) ready to be deployed.

This is a 3 part blog series. We will see how to deploy a ML model as a microservice. We will see 3 different ways of doing the same using:

1. [Flask]() 
2. [Falcon]()
3. [Jupyter notebook as service]() (Wow!)


### Microservice - what and why ?

Microserivce is an architecture pattern. It can best be thought as being completely opposite of Monolithic architecture ([problems](https://www.thoughtworks.com/insights/blog/monoliths-are-bad-design-and-you-know-it)). The central idea is to break the system/applications into small chunks(services) based on functionality. Each chunk does a specicifc job and does only that. These services talk to each other using HTTP/REST (synchronous or asynchronous). Want to take a deep dive ? I suggest read [quora answer](https://www.quora.com/What-is-Microservices-Architecture) and [Martin Fowler's article](https://www.martinfowler.com/articles/microservices.html).


Part 1 - Flask
Part 2 - Falcon
Part 3 - Notebook

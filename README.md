# README
# Project 1: Navigation
## Udacity Deep Reinforcement Learning Nanodegree


## Introduction

This repository shows how `Project 1: Navigation` from the Udacity Deep Reinforcement Learning Nanodegree was tackled.

The README provides some general information. The repository also contains source code (in the `src` directory) and a report (in the `report` directory).

## Project details

![banna](img/visualization_1.gif)

_(Udacity)_

The goal of the project is to teach a Deep Reinforcement Learning agent to intercept yellow bananas (reward +1) while avoiding blue bananas (reward -1) as it moves through an enclosed two-dimensional square environment. The visualized perspective is first-person. 

The environment is driven by the [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). However, the environment is not identical to the Unity-included 'Banana Collector' environment.

The state-space is 37-dimensional, containing measurements such as the agent's velocity and ray-based perception of objects in the region ahead of the agent.

At each step, the agent can choose between four actions:

* 0: forward
* 1: backward
* 2: left
* 3: right

Learning is episodic. Each episode always contains exactly 300 steps.

The problem is considered to be solved as soon as the average reward over the preceding 100 episodes exceeds 13.0. It is known that the problem should be solvable within 1800 episodes. The fewer episodes required, the better.

## Setup

The following steps will create the computing environment that was used for training.

1. On AWS, spin up a p2.xlarge instance in the N. Virginia region using the Udacity AMI `ami-016ff5559334f8619`.
2. Once the instance is up and running, SSH into the instance.
3. Run the following commands to clone the appropriate Udacity repository and install some Python dependencies:

	```
	conda activate pytorch_p36

	cd ~
	mkdir -p external/udacity

	cd ~/external/udacity
	git clone https://github.com/udacity/deep-reinforcement-learning.git

	cd ~/external/udacity/deep-reinforcement-learning/python
	pip install .
	pip install seaborn
	```

4. Download resources required for the environment:

	```
	cd ~/external/udacity/deep-reinforcement-learning/p1_navigation
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip
	unzip Banana_Linux_NoVis.zip
	```
	
	Note that for this step, 'no visualisation'-flavoured resources are used.
	
5. To correct an error in Jupyter Notebook/Lab that occurs in this computing environment as of 2019-07, perform the following:

	```
	pip install 'prompt-toolkit<2.0.0,>=1.0.15' --force-reinstall
	```
	
6. Copy the files in the `src` directory of this repository to the `~/external/udacity/deep-reinforcement-learning/p1_navigation/src` directory on the EC2 instance.	
	
7. Deactivate the conda environment:

	```
	conda deactivate
	```
	
8. Securely set up Jupyter:

	```
	cd ~/
	mkdir ssl
	cd ssl
	openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout "cert.key" -out "cert.pem" -batch	
	
	jupyter notebook --generate-config
	jupyter notebook password  # Enter and verify a password.
	
	```
	
9. Using an editor, add the following to the top of the `~/.jupyter/jupyter_notebook_config.py` file:

	```
	c = get_config()
	c.NotebookApp.certfile = u'/home/ubuntu/ssl/cert.pem'
	c.NotebookApp.keyfile = u'/home/ubuntu/ssl/cert.key'
	c.IPKernelApp.pylab = 'inline'
	c.NotebookApp.ip = '*'
	c.NotebookApp.open_browser = False	
	```
	
10. Start Jupyter Lab:

	```
	cd
	jupyter lab	
	```
	
11. If the EC2 security group that's in force allows traffic to port 8888, point a local browser to https://[ec2-ip-address]:8888. Otherwise execute the following in a _local_ terminal:

	```
	sudo ssh -L 443:127.0.0.1:8888 ubuntu@[ec2-ip-address]
	```
	
	Then point a local browser to https://127.0.0.1.

## Training

To replicate training, navigate to `~/external/udacity/deep-reinforcement-learning/p1_navigation/src` within Jupyter Lab. Open the `train.ipynb` notebook and follow the steps therein.

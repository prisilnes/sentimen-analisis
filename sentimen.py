# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:46:15 2020

@author: Ines
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re #library for using RegEx
import nltk #library for various NLP toolkits

import heapq

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score

# Splitting training and testing set
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

def hello():
    print "hello"

hello()

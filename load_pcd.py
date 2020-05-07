#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:09:39 2020

@author: abel
"""

import pickle
import numpy as np

dbfile = open('coords', 'rb')      
db = pickle.load(dbfile)
print(db.shape) 

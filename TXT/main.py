# author: Syndicate 7
# Project: Text and Web Based Analytics

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
import string

from data_processing import preprocessing


def main():
    '''
    Step 1: Run the pre-processor
    Step 2: Run different classifiers based off the input
    TODO: Add parameters to pass via the main funciton, eg: Kernel function in SVM

    Step 3: Output the results of these classifiers

    TODO: Find a way to represent the output on some sort of scale

    --------------------
    Step 4: Scrape the web-data for information about the companies
    Step 5: Map the companies with our best classifier
    Step 6: Activate Voice Recorder or in turn collect free text
    Step 7: Display the results in terms of the users personality scale and also best cultural fit

    TODO: Alternatively we could just focus on the Big 4 firms and see which candidates are the best match

    :output: the summary of statistics of the classifiers
    '''

    # Step 1
    df = preprocessing(False)




if __name__ == "__main__":
    main()

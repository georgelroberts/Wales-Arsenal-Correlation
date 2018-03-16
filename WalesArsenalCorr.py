# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 18:58:48 2018

@author: George

Match whether Arsenal and Wales both played on a certain weekend (Friday,
Saturday, Sunday, Monday).


"""

import pandas as pd
import os
import scipy.stats as st
import numpy as np

os.chdir('C:\\Users\\George\\OneDrive - University Of Cambridge\\Others\\Machine learning\\Wales Arsenal Correlation')
walesData = pd.read_csv('WalesData.csv')
walesData['Date'] = pd.to_datetime(walesData['Date'])
walesData = walesData.drop('Match report', axis=1)
dateFrom = 2015

arsenalData = pd.read_csv('ArsenalData.csv')
arsenalData['date'] = pd.to_datetime(arsenalData['date'], format='%d/%m/%Y')


walesData.fillna(method='ffill', inplace=True) # Only the first value of competition name is present
walesData = walesData[walesData['Score'] != '\xa0? –\xa0?'] # Remove matches that haven't happened yet
walesData = walesData[walesData['Date'].dt.year >= dateFrom]
walesData = walesData[walesData['Date'].dt.dayofweek>=4] # Keep only weekends


def replaceCompetitions(comp):
    """Comptetitions in the data have their year written in, so
    standardise this """
    comp = comp.lower()
    if 'six nations' in comp:
        comp = 'Six Nations'
    elif 'internationals' in comp:
        comp = 'Internationals'
    elif 'tour' in comp:
        comp = 'Tour'
    elif 'world cup' in comp:
        comp = 'World Cup'
    return comp

walesData['Competition'] = walesData['Competition'].apply(replaceCompetitions)
walesData.dropna(axis=1, inplace=True, how='all')
walesData.dropna(axis=0, inplace=True)

def walesResults(dat):
    """Create a dataframe column defining whether wales won, lost or drew
    """
    dat = dat.lower()
    if 'draw' in dat:
        res = 2
    elif 'wales' in dat:
        res = 1
    else:
        res = 0
    return res
walesData['walesWin'] = walesData['Winner'].apply(walesResults)

def dateTimetoWeekend(dateList, startYear):
    """ Remove Monday-Thursday and then hash each weekend so that each
    year will have a separate value """
    weekHash = dateList.dt.week + 52 * (dateList.dt.year - startYear)
    return weekHash

walesData['weekHash'] = dateTimetoWeekend(walesData['Date'], dateFrom)

walesResults = walesData[['Date', 'weekHash', 'walesWin']]

# Two possibilities: Correlation when they are on the same day and correlation
# when they are on the same weekend. The former is easy, the latter requires
# removing all days that aren't a Friday, Saturday or Sunday.

arsenalData.dropna(axis=1, inplace=True, how='all')
arsenalData.dropna(axis=0, inplace=True)
arsenalData = arsenalData[arsenalData['date'].dt.year >= dateFrom]
arsenalData = arsenalData[arsenalData['date'].dt.dayofweek>=4] # Keep only weekends

arsenalData = arsenalData[arsenalData.Results.str[0].str.isnumeric()]

def arsenalResults(dat):
    """Arsenal have never had a game with a double digit scoreline, so just
    take the first and third string locations"""
    arsScore = int(dat[0])
    othScore = int(dat[2])
    if arsScore > othScore:
        res = 1
    elif arsScore == othScore:
        res = 2
    else:
        res = 0
    return res

arsenalData['Winner'] = arsenalData['Results'].apply(arsenalResults)

arsenalData.columns = ['Round', 'Date', 'KO', 'Place', 'Opponent','Score', 'Competition', 'arseWin']
arsenalData['weekHash'] = dateTimetoWeekend(arsenalData['Date'], dateFrom)

arseResults = arsenalData[['Date', 'weekHash', 'arseWin']]

mergedResults = walesResults.merge(arseResults, on='weekHash')
mergedResults = mergedResults[mergedResults.arseWin != 2]
mergedResults = mergedResults[mergedResults.walesWin != 2]

corrs = st.pearsonr(mergedResults['walesWin'], mergedResults['arseWin'])


def absoluteCounts(merged, dateFrom):
    """ Prints the amount of times arsenal and wales have won together, lost
    together, and got different results """
    same = merged[merged['walesWin'] == merged['arseWin']]
    different = merged[merged['walesWin'] != merged['arseWin']]
    totalGames = merged.shape[0]
    together = same.shape[0]
    togetherW = same.walesWin.value_counts().loc[1]
    togetherL = together - togetherW

    print('Arsenal and Wales have played {}'.format(totalGames) +
          ' games on the same weekend since {}.'.format(dateFrom))
    print('Of these games, they have had the same result {}'.format(together) +
          ' times, with {} wins and {} losses'.format(togetherW, togetherL))


absoluteCounts(mergedResults, dateFrom)

print('The Pearson R value is {:.2f} and the p-value is {:.2f}'.format(
        corrs[0], corrs[1]))

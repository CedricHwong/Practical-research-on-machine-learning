import bayes
import re


emailText = open('email/ham/6.txt').read()
regEx = re.compile('\\W*')
listOfTokens = regEx.split(emailText)
bayes.spamTest()
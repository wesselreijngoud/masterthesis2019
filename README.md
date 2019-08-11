This contains all the files to reproduce the results from my thesis research. 

Except for the word embeddings file w2v.bin which is the same as used in Van der Goot and Van Noord (2017)  [https://arxiv.org/pdf/1710.03476.pdf]


To run the baselines:
```
python3 freqbaseline.py ./Data/Owoputi/owoputi.train+dev ./Data/Owoputi/owoputi.test -r

python3 simplebaseline.py ./Data/Owoputi/owoputi.train+dev ./Data/Owoputi/owoputi.test -r 
```
To run the classifier for results:
```
python3 classifier.py ./Data/Owoputi/owoputi.train+dev ./Data/Owoputi/owoputi.test -r
```

T run the classifier for error analysis:
```
python3 classifier.py ./Data/Owoputi/owoputi.train+dev ./Data/Owoputi/owoputi.test -a 
```


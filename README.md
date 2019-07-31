This contains all the files to reproduce the results from my thesis research. 

Except for the word embeddings file w2v.bin which can be found on the rug server in /net/shared/rob/wessel 


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


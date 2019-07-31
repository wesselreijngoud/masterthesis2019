

file2 = open("lexnorm2015.data.fixed2", "r")
file3 = open("owoputiFinal2.final")
blist = []
dlist = []
alist= []
for line in file2:
    if len(line) == 2:
        blist.append(0)
    if len(line) > 1:
    	dlist.append(1)

clist=[]
for line in file3:
    if len(line) == 1:
        clist.append(0)
    if len(line) > 1:
    	alist.append(1)

print("Nr. Of Tweets lexnorm:", len(blist), len(dlist), (len(dlist) - 2950))
print("Nr. Of Tweets owoputi", len(clist), len(alist))
# print("Human unique:", len(set(blist)))
file2.close()

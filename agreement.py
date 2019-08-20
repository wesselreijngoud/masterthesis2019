import sys

if len(sys.argv) < 3:
    print('give 2 annotated filesi, eg. data/pairsAnnotated, data/secondAnnotator')
    exit()


def getConversions(filename):
    conversions = {}
    for line in open (filename):
        splitted = line.split()
        splitted = [x.lower() for x in splitted]
        orig = splitted[0].lower()
        cor = splitted[-2].lower()
        if len(splitted) > 3:
            cor = ' '.join(splitted[2:-1])
        if  '->' in cor:
            cor = ''
        conversions[orig + ' ' + cor] = int(splitted[-1])
    # print(conversions)
    return conversions

def getConversionsFront(filename):
    conversions = {}
    for line in open (filename):
        splitted = line.split()
        splitted = [x.lower() for x in splitted]
        cat  = int(splitted[0])
        orig = splitted[1].lower()
        cor = splitted[-1].lower()
        if len(splitted) > 4:
            cor = ' '.join(splitted[3:-1]) + ' ' + splitted[-1]
        if  '->' in cor:
            cor = ''
        conversions[orig + ' ' + cor] = cat
    # print(conversions)
    return conversions

mine = getConversions(sys.argv[1])
rik = getConversionsFront(sys.argv[2])

total = 0
agr = 0
fileAll = open('agreement.all', 'w')
fileDiff = open('agreement.diff', 'w')
fileDiffTxt = open('agreement.diff.txt', 'w')
for conv in rik:
    total += 1
    try:
        fileAll.write(str(mine[conv]) + '\t' + str(rik[conv]) + '\n')
    except KeyError:
        continue

    if  rik[conv] == mine[conv]:
        agr += 1
        continue
    else:
        fileDiff.write(str(mine[conv]) + '\t' + str(rik[conv]) + '\n')
        fileDiffTxt.write(str(mine[conv]) + '\t' + str(rik[conv]) + '\t' + conv + '\n')
        if mine[conv] == 2 and rik[conv] == 1:
            #print(conv) 
            pass
        else:
            #print(mine[conv], rik[conv], conv)
            pass
fileAll.close() 
fileDiff.close()  
fileDiffTxt.close()

print(agr/total)

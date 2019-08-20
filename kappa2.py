
import sys
#Program that calculates Cohen Kappa annotator agreement

kappatest = open(sys.argv[1])

def kappa(judge1,judge2):
	#definitions
	total = len(judge1)
	agreement = 0
	kappa = 0
	pE = 0
	judge1list = []
	judge2list = []
	valuelist = []

	#get distinct values from judgelist
	for item in judge1:
		if item not in valuelist:
			valuelist.append(item)

    #calculate agreement
	for x in range(len(judge1)):
		if judge1[x] == judge2[x]:
			agreement +=1
		
	#probability that judges agree per judge	
	for y in valuelist:
		judge1list.append(judge1.count(y)/total)
		judge2list.append(judge2.count(y)/total)
	
	#calculate pE
	for x in range(len(valuelist)):
		pE+=((judge1list[x]*judge2list[x]))

	#calculate pA
	pA = agreement / total
	#calculate kappa
	kappa = (pA-pE)/(1-pE)

	print('Kappa:',round(kappa,4))


def main():
    judge1 = []
    judge2 = []
    for judges in kappatest:
        x1,x2 = judges.split('\t')
        judge1.append(x1.rstrip())
        judge2.append(x2.rstrip())

    kappa(judge1,judge2)

main()

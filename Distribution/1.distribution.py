import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import myutils

if len(sys.argv) < 2:
    print('please give annotated corpus as arg, example:')
    print('python3 scripts/1.distribution.py data/lexnorm2015.annotated')
    exit(1)

plt.style.use('rob.mplstyle')
fig, ax = plt.subplots(figsize=(8,5), dpi=300)

catCounts = [0] * 16
catUniq = [0] * 16
seen = set()
total = 0
totalUniq = 0
for line in open(sys.argv[1]):
    splitted = line.split()
    if len(splitted) < 2:
        continue
    total += 1
    cat = int(splitted[0])
    if len(splitted) > 3:
        repl = splitted[1] + ':' + splitted[3]
    else:
        repl = splitted[1]

    if repl not in seen:
        seen.add(repl)
        catUniq[cat] += 1
        totalUniq += 1
    catCounts[cat] += 1 

# FOR PERCENTAGES
#for i in range(len(catCounts)):
#    catCounts[i] = (catCounts[i] * 100) / total
#    catUniq[i] = (catUniq[i] * 100) / totalUniq

barwidth = 1/3
idxs = []
idxs2 = []
for i in range(len(catCounts)-1):
    idxs.append(1/6  + i + barwidth * .5)
    idxs2.append(1/6 + i + barwidth * 1.5)

myutils.setTicks(ax, myutils.cats, 45)
ax.bar(idxs, catCounts[1:], width= barwidth, color = myutils.colors[0], label='Total')
ax.bar(idxs2, catUniq[1:], width=barwidth, color=myutils.colors[1], label='Repl. Types')



"""EDIT 100-NUMBERS BENEATH HERE TO EDIT THE YLIMIT RANGE"""
ax.plot([5,5], [0, 840], color='black', linestyle='dashed', clip_on=False)
ax.text(2, 820, 'Unintended', fontsize=15)
ax.text(5.3, 820, 'Intended', fontsize=15)


ax.set_ylabel("Number of replacements", color='black')
#ax.set_xlabel("Category", color='black')
leg = plt.legend(loc='upper left')
leg.get_frame().set_linewidth(1.5)
ax.set_ylim(0,800)
ax.set_xlim(0,len(catCounts) - 1)
fig.savefig('distribution.pdf', bbox_inches='tight')
#plt.show()


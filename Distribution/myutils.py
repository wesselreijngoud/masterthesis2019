import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('rob.mplstyle')
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] 
colors = colors + colors
featNames = ['Embeddings', 'Aspell', 'Lookup', 'Word.*', 'Split', 'N-grams', 'Dictionary', 'CharOrder', 'Length', 'Cont.Alpha', 'OrigWord']
cats = ['Typo', 'Missing apo.', 'Spelling err.', 'Split', 'Merge', 'Phrasal abbr.', 'Repetition', 'Short. vow.', 'Short end', 'Short other', 'Reg. trans', 'Other trans.', 'Slang', 'Unk', 'Inf. Contract.']
langs = ['nl', 'es', 'en2', 'en1', 'sl', 'hr', 'sr']
corpora = ['GhentNorm', 'TweetNorm', 'LexNorm1.2', 'LexNorm2015', 'Janes-Norm', 'ReLDI-hr', 'ReLDI-sr']
sizes = [769, 566, 2576, 2950, 6227, 6350, 5518]

def setTicks(ax, labels, rotation = 0):
    ticks = []
    for i in range (len(labels)):
        ticks.append(i + .55)
        if rotation == 0:
            ticks[-1]+=.1

    ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(len(labels)+1))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(ticks))

    ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.FixedFormatter(labels))

    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('right')
        tick.label1.set_rotation(rotation)


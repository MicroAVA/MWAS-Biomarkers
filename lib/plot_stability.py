import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
import numpy as np

# activate latex text rendering
rc('text', usetex=True)
cirrhosis = pd.read_csv('output/cirrhosis-KI.csv', index_col=0)
t2d = pd.read_csv('output/t2d-KI.csv', index_col=0)
obesity = pd.read_csv('output/obesity-KI.csv', index_col=0)
fig, axes = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(5, 5))
axes[0].set_yticks(np.arange(0, 1.0, step=0.2))

cirrhosis.plot(ax=axes[0], style=['g.--', 'y.--', 'b.--', 'r*-'], legend=False)
axes[0].set_title('Cirrhosis')

axes[1].set_ylabel('Stability')
axes[0].set_xlabel('Features left(%)')

t2d.plot(ax=axes[1], style=['g.--', 'y.--', 'b.--', 'r*-'], legend=False)
axes[1].set_title('T2D')
axes[1].set_xlabel('Features left(%)')

obesity.plot(ax=axes[2], style=['g.--', 'y.--', 'b.--', 'r*-'], legend=False)
axes[2].set_title('Obesity')
axes[2].set_xlabel('Features left(\\%)')

plt.legend(loc='center left', bbox_to_anchor=(1.05, 2), labelspacing=1.5,
           labels=['mRMR', 'SVM-RFE', 'ReliefF', '\\textbf{DF}'])
plt.subplots_adjust(hspace=0.5)
plt.savefig('output/stability.png', bbox_inches='tight')
plt.show()

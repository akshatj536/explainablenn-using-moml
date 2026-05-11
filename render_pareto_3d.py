import base64
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Base64-encoded float64 arrays extracted from pareto_3d.html
x_b64 = "Y+F427OZ7D+h98fFe0rsPwqZ1xzVZew/p76rxLNO7D8ph2egPYvrP1Amtz2pwus/FiK7xhbu6z98MpCeMxTsP5yAp3ra9es/b6D1SEZ66z9p2Cewe2nrP20GNIr1g+s/Q9mLGAXB6z8DKqGWXtPrPwPQ6a8rS+s/dI0rOjw+6z/UG7deyE7rPxnbO4d/des/NO5odj5f6z9kSF5t/7HqP6Yzr88vIOs/RO5P9qfu6j8qsaW1CkvrP7OVmPC7Lus/qbBDBu836z9pP19Znm3qP7Zxne+h3+o/Y7Bc1D0X6z+ZT6AoENTqP3YhgOp6qOo/hckNMCwi6T9NiEhg6hXpP+lYvUMiHek/ib80bd8j6T8lGk+0/AjpP4AfLybOE+k/IJO1/DLt6D+A8rlD0gfpPwT+mi5T++g/MytjiS/h6D/0u0gL1SDrP9+nVfaIIes/t9DjY+W56z8AlCKg/+zrP5EOT9ipT+w/8cLlNmgz7D/fnElMwvLrP+mBnimkkes/9LtIC9Ug6z/0u0gL1SDrPwCUIqD/7Os/8cLlNmgz7D/fnElMwvLrP9+cSUzC8us/6YGeKaSR6z/0u0gL1SDrP9+cSUzC8us/hkqK95jf6z/pgZ4ppJHrP+mBnimkkes/ORfYuppl6z85F9i6mmXrPw=="
y_b64 = "eNzZsRGAxT81WfZcjXrDPw5nsNZ+08Q/hqBQI+ppxD9m/VZWby68P5Dyj0XkpMA/3MXNLGZewj/U2LKkmYrCP6hJXT3HXcM/8kC1FERXuT+gipfChfC3P7uP/bD6abw/FrXzzVVmwT/b9paLUajCPzEbIhQdCrc/wsMXHUiStj905vKuuPm8P4vEnXXk/cA/SPCApQNFwT+XDsPn+P+IPwBXvPJw35g/BG6YKEWqrz9+byIimPu2P0lzSgLPL7w/JMT334y3vD+QivPe1liFP4cClSVJo5k/d2u7ZL/Apz8tr7ve/DWvP2GiNaB+A4w/eAiuofNsxD+MNmc3hpq7P7sHcbZAAsI/rAd+yqortT9u+aFfnR+4P/8dwzSEZqQ/K2N3wUYntD8oIMvpyG2jP0THs2tox5c/3gvQ07y0fD+WF4BWCsiZP14ZzSs95Zo/YSSZetyRwD9DDg2gaK/BPo6nETi5dcQ/XOYqL6AzxD9zbc2wzFbDP7T47VR84ME/lheAVgrImT+WF4BWCsiZP0MODaBor8E/XOYqL6AzxD9zbc2wzFbDP3NtzbDMVsM/tPjtVHzgwT+WF4BWCsiZP3NtzbDMVsM/xBXHRrMrwz+0+O1UfODBP7T47VR84ME/tx4UqRVlwT+3HhSpFWXBPw=="
z_b64 = "wE6qd9uwuT9A74fiA/S9PyBJ5Wbedrc/qBo4C2Z3tD+k6AC+UhDLPyDF2o0ZjMc/MKzDna/AwT/4dPkQdCW5P3CE4vvQu7A/1MnXmB0Zxz9oenGizwe/P4Dz6kQyWr4/2JZKBr2msz/gqjS2u0OvP9AF/OCwfsE/GL5xUN0Htz/wdacqJOSyP6Bdr01HzKo/gH7Q7DlSoD9Y0koeGd3GPwhAMX8ZvL8/gMXYOToytj+YQ2+EmVizP2Ag0T7da6c/UIxky6d+pT/YSyTyVoPDPxCNwHz9fLs/0LDFeow9tz8Ys+rHk0W0P4Bb7QL5Gbs/wHbcgzfXlD+gllA0QgCTPwDbWXQRCYU/AI+qgi74nT8AwGqvWj8OP8CSwJPfCJI/AIDIh+NC9D4AwOnQC5IVPwCwW/4RpyQ/ADJg3GWYbT94uk0kzFnAP0g5KPEy1b4/jgOyQF1Z0D+I0Kv3JdfKP+iIdMa9jLQ/oJymUHWPsz+Q6R3GTwixP6Abe5FLvKY/eLpNJMxZwD94uk0kzFnAP4jQq/cl18o/oJymUHWPsz+Q6R3GTwixP5DpHcZPCLE/oBt7kUu8pj94uk0kzFnAP5DpHcZPCLE/uNGQ/K9wsD+gG3uRS7ymP6Abe5FLvKY/gJxAeRNEnj+AnEB5E0SePw=="

def decode(b64):
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype='<f8')

x = decode(x_b64)
y = decode(y_b64)
z = decode(z_b64)

# Knee point
kx, ky, kz = 0.8465870998093233, 0.04639242272033688, 0.0907829093682182

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z, c=x, cmap='viridis', s=40, alpha=0.85,
                depthshade=True, label='Pareto Front', zorder=2)
ax.scatter([kx], [ky], [kz], c='red', s=200, marker='D',
           label='Knee Point', zorder=3, edgecolors='darkred', linewidths=1.2)

cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.1)
cbar.set_label('Utility (macro F1)', fontsize=10)

ax.set_xlabel('Utility (macro F1)', fontsize=11, labelpad=10)
ax.set_ylabel('Trust Gap (ACE)', fontsize=11, labelpad=10)
ax.set_zlabel('Equity Gap\n(|F1_clean - F1_noisy|)', fontsize=10, labelpad=10)
ax.set_title('Pareto Front', fontsize=14, fontweight='bold', pad=15)

ax.view_init(elev=25, azim=135)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgrey')
ax.yaxis.pane.set_edgecolor('lightgrey')
ax.zaxis.pane.set_edgecolor('lightgrey')
ax.grid(True, linestyle='--', alpha=0.4)

ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig('pareto_3d.png', dpi=150, bbox_inches='tight')
print("Saved pareto_3d.png")

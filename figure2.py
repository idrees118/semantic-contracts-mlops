import matplotlib.pyplot as plt
import numpy as np

categories = ['Aggregation', 'Quality', 'Scaling', 'Structural', 'Temporal']
validators = ['Semantic', 'Ensemble', 'KS Drift', 'GE']

grid = np.array([
    [100, 83, 100, 100, 100],   # Semantic:  1/1, 5/6, 6/6, 9/9, 6/6
    [100, 17,  83,  56,  33],   # Ensemble:  1/1, 1/6, 5/6, 5/9, 2/6
    [100, 17,  83,  56,   0],   # KS Drift:  1/1, 1/6, 5/6, 5/9, 0/6
    [100, 17,  50,  56,  33],   # GE:        1/1, 1/6, 3/6, 5/9, 2/6
])

fig, ax = plt.subplots(figsize=(6.8, 4.2))

im = ax.imshow(grid, cmap='Reds', vmin=0, vmax=100)

ax.set_xticks(np.arange(len(categories)))
ax.set_yticks(np.arange(len(validators)))
ax.set_xticklabels(categories, fontsize=9)
ax.set_yticklabels(validators, fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        val = grid[i, j]
        color = 'white' if val >= 50 else 'black'
        ax.text(j, i, f'{val:.0f}%',
                ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)

cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
cbar.ax.tick_params(labelsize=9)
cbar.set_label('Detection Rate (%)', fontsize=9)

ax.set_title('Detection Performance by Mutation Category (%)',
             fontsize=10, pad=6)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig('fig2_red_ieee.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig2_red_ieee.png', dpi=300, bbox_inches='tight')

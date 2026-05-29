import matplotlib.pyplot as plt
import numpy as np

labels = ['Semantic\nValidator', 'Ensemble\nStatistical', 'KS Drift\n(original)', 'GE\nComprehensive']
drs = [96.4, 49.1, 42.0, 42.9]
# Asymmetric Wilson CI half-widths from Table III
yerr_lower = [14.1, 9.1, 8.8, 8.8]
yerr_upper = [3.0, 9.1, 9.2, 9.2]

# 🔥 Professional, balanced palette (not flashy, not biased)
colors = ['#355070',  # muted navy
          '#6D597A',  # soft purple-gray
          '#B56576',  # muted rose
          '#8D99AE']  # cool gray-blue

fig, ax = plt.subplots(figsize=(6.2, 4))

bars = ax.bar(labels, drs,
              color=colors,
              edgecolor='black',
              linewidth=0.6)

# Error bars — drawn explicitly so caps are always visible above bar top
for i, (label, dr, lo, hi) in enumerate(zip(labels, drs, yerr_lower, yerr_upper)):
    # Upper whisker only (visible above bar)
    ax.errorbar(i, dr, yerr=[[0], [hi]],
                fmt='none', ecolor='black', capsize=4, linewidth=1.0)
    # Lower CI annotation as a horizontal tick mark below bar top
    # Lower whisker below bar
    ax.errorbar(i, dr, yerr=[[lo], [0]],
                fmt='none', ecolor='black', capsize=4, linewidth=1.0)

# Value labels with CI range shown below each bar label
for i, (bar, val, lo, hi) in enumerate(zip(bars, drs, yerr_lower, yerr_upper)):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + hi + 1.2,
            f'{val:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2,
            -8,
            f'[{val-lo:.1f}, {val+hi:.1f}]',
            ha='center',
            va='top',
            fontsize=7,
            color='#444444',
            transform=ax.transData)

# Limits & labels
ax.set_ylim(-11, 108)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel('Detection Rate (%)', fontsize=10)

# Title (smaller, cleaner)
ax.set_title('Pooled Detection Rate with 95% Wilson Confidence Intervals\n(112 Trials)',
             fontsize=10, pad=6)

# Remove clutter
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig1_detection_rates.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig1_professional_colors.png', dpi=300, bbox_inches='tight')
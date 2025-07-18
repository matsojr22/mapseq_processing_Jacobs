import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

# Set font preferences for editable SVG text
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = ['Helvetica', 'Arial', 'sans-serif']

# Load the TSV file
df = pd.read_csv("Summary_data_07012025 - Copy of HAN_ALL_combined.tsv", sep='\t')

# Define the brain areas of interest
areas = ['RSP', 'PM', 'AM', 'AL', 'LM']
umisum_cols = [f'UMISum_{area}' for area in areas]
projcount_cols = [f'ProjCount_{area}' for area in areas]

# Extract and sort UMI data
df_comp = df[['Age'] + umisum_cols].copy()
df_comp['Age'] = pd.Categorical(df_comp['Age'], categories=['p3', 'p12', 'p20', 'p60'], ordered=True)
df_comp = df_comp.sort_values('Age')

# Normalize UMI counts per row to percentage
df_percent = df_comp.set_index('Age')
df_percent = df_percent.div(df_percent.sum(axis=1), axis=0) * 100

# Plot UMI composition
fig, ax = plt.subplots(figsize=(8, 3))
colors = plt.cm.tab10.colors[:len(areas)]

bottom = [0] * len(df_percent)
for i, area in enumerate(areas):
    values = df_percent[f'UMISum_{area}']
    ax.barh(df_percent.index, values, left=bottom, color=colors[i], label=area)
    bottom = [sum(x) for x in zip(bottom, values)]

ax.set_xlim(0, 100)
ax.set_xticks([])
ax.set_yticks(range(len(df_percent)))
ax.set_yticklabels(df_percent.index)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(False)

legend_elements = [Patch(facecolor=colors[i], label=area) for i, area in enumerate(areas)]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.tight_layout()
plt.savefig("UMI_composition_by_age.svg", format='svg')
plt.close()

# Extract and sort ProjCount data
df_proj = df[['Age'] + projcount_cols].copy()
df_proj['Age'] = pd.Categorical(df_proj['Age'], categories=['p3', 'p12', 'p20', 'p60'], ordered=True)
df_proj = df_proj.sort_values('Age')

# Normalize projection counts per row to percentage
df_proj_percent = df_proj.set_index('Age')
df_proj_percent = df_proj_percent.div(df_proj_percent.sum(axis=1), axis=0) * 100

# Plot ProjCount composition
fig, ax = plt.subplots(figsize=(8, 3))

bottom = [0] * len(df_proj_percent)
for i, area in enumerate(areas):
    values = df_proj_percent[f'ProjCount_{area}']
    ax.barh(df_proj_percent.index, values, left=bottom, color=colors[i], label=area)
    bottom = [sum(x) for x in zip(bottom, values)]

ax.set_xlim(0, 100)
ax.set_xticks([])
ax.set_yticks(range(len(df_proj_percent)))
ax.set_yticklabels(df_proj_percent.index)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(False)

legend_elements = [Patch(facecolor=colors[i], label=area) for i, area in enumerate(areas)]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.tight_layout()
plt.savefig("ProjCount_composition_by_age.svg", format='svg')
plt.close()
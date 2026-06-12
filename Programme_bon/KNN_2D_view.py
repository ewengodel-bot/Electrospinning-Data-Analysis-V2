import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# 1. Load data and force numeric types
data = pd.read_csv("resultats_complets.csv", sep=',', decimal='.')
for col in ['Distance', 'Flow-rate', 'Voltage', 'Classe']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

X = data[['Distance', 'Flow-rate', 'Voltage']]
y = data['Classe']

# 2. Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# 4. New point to predict
new_setting = pd.DataFrame([[13, 0.7, 15]], columns=['Distance', 'Flow-rate', 'Voltage'])
new_setting_scaled = scaler.transform(new_setting)
prediction = knn.predict(new_setting_scaled)

# Professional color code
colors_dict = {
    1: '#2ca02c',  # Green
    2: '#ffc107',  # Yellow
    3: '#ff7f0e',  # Orange
    4: '#d62728'   # Red
}

# ==========================================
# --- 3D PLOT ---
# ==========================================
fig = plt.figure(figsize=(11, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

for cl in sorted(y.unique()):
    mask = (y == cl)
    ax.scatter(X.loc[mask, 'Distance'], X.loc[mask, 'Flow-rate'], X.loc[mask, 'Voltage'],
               color=colors_dict.get(int(cl), 'grey'), s=50, alpha=0.9, edgecolors='w', label=f'Class {int(cl)}')

ax.scatter(new_setting['Distance'], new_setting['Flow-rate'], new_setting['Voltage'],
           color='blue', marker='*', s=160, edgecolor='black', linewidth=1.2,
           label=f'New Point (Predicted: Class {prediction[0]})')

ax.set_xlabel('Distance (mm)', labelpad=10)
ax.set_ylabel('Flow-rate (mL/h)', labelpad=10)
ax.set_zlabel('Voltage (kV)', labelpad=10)
ax.set_title('KNN Classification - 3D View', fontsize=14, fontweight='bold')
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('knn_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# --- 2D PROJECTIONS (OPAQUE SURFACE) ---
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

axes = [ax1, ax2, ax3]
views = [('Distance', 'Flow-rate'), ('Distance', 'Voltage'), ('Flow-rate', 'Voltage')]
titles = ['Top View (Distance vs Flow-rate)', 'Front View (Distance vs Voltage)', 'Side View (Flow-rate vs Voltage)']

# Create a combined dataframe to easily sort and filter
df_plot = X.copy()
df_plot['Class'] = y

for i in range(3):
    x_col, y_col = views[i]

    # Logic to only keep the "closest" point to the viewer for each 2D coordinate
    if i == 0:   # Top View: we see the highest Voltage
        df_surface = df_plot.sort_values(by='Voltage', ascending=False).drop_duplicates(subset=[x_col, y_col])
    elif i == 1: # Front View: we see the lowest Flow-rate
        df_surface = df_plot.sort_values(by='Flow-rate', ascending=True).drop_duplicates(subset=[x_col, y_col])
    elif i == 2: # Side View: we see the highest Distance
        df_surface = df_plot.sort_values(by='Distance', ascending=False).drop_duplicates(subset=[x_col, y_col])

    # Plot the filtered surface points
    for cl in sorted(df_surface['Class'].unique()):
        mask = (df_surface['Class'] == cl)
        axes[i].scatter(df_surface.loc[mask, x_col], df_surface.loc[mask, y_col],
                        color=colors_dict.get(int(cl), 'grey'), s=60, alpha=0.9, edgecolors='w',
                        label=f'Class {int(cl)}')

    # Plot the new point on top
    axes[i].scatter(new_setting[x_col], new_setting[y_col],
                    color='blue', marker='*', s=160, edgecolor='black', zorder=10)

    axes[i].set_xlabel(x_col)
    axes[i].set_ylabel(y_col)
    axes[i].set_title(titles[i])

# Single clean legend at the bottom
handles, labels = ax1.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.0),
           fontsize=11)

plt.suptitle('2D Projections (Surface view) & New Point (*)', fontsize=15,
             fontweight='bold', y=0.95)
plt.tight_layout(rect=[0, 0.12, 1, 0.95])
plt.savefig('knn_projections_2d_surface.png', dpi=300, bbox_inches='tight')
plt.show()
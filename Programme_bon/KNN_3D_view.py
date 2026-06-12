import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Force plot to open in the web browser
pio.renderers.default = 'browser'

# 1. Load and clean data
data = pd.read_csv("resultats_complets.csv", sep=',', decimal='.')
for col in ['Distance', 'Flow-rate', 'Voltage', 'Classe']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

X = data[['Distance', 'Flow-rate', 'Voltage']]
y = data['Classe']
data['Class_Str'] = 'Class ' + data['Classe'].astype(int).astype(str)

# --- STEP 2: NORMALIZATION ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train the KNN model on scaled data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# 3. New point to predict
new_setting = pd.DataFrame([[12.5, 0.7, 18]], columns=['Distance', 'Flow-rate', 'Voltage'])
new_setting_scaled = scaler.transform(new_setting)

prediction = knn.predict(new_setting_scaled)
predicted_class = int(prediction[0])

# Find the indices of the 3 nearest neighbors for the plot
neighbors_indices = knn.kneighbors(new_setting_scaled, return_distance=False)[0]

# --- 4. CREATE INTERACTIVE 3D PLOT ---
fig = go.Figure()

# Professional colors matching your slides
colors_dict = {
    'Class 1': '#2ca02c', # Green
    'Class 2': '#ffc107', # Yellow
    'Class 3': '#ff7f0e', # Orange
    'Class 4': '#d62728'  # Red
}

# Add standard dataset points
for class_name in sorted(data['Class_Str'].unique()):
    df_class = data[data['Class_Str'] == class_name]
    fig.add_trace(go.Scatter3d(
        x=df_class['Distance'], y=df_class['Flow-rate'], z=df_class['Voltage'],
        mode='markers', name=class_name,
        marker=dict(size=6, color=colors_dict.get(class_name, 'grey'), line=dict(width=1, color='white'), opacity=0.8)
    ))

# STEP 3: Highlight the 3 real nearest neighbors
df_neighbors = data.iloc[neighbors_indices]
fig.add_trace(go.Scatter3d(
    x=df_neighbors['Distance'], y=df_neighbors['Flow-rate'], z=df_neighbors['Voltage'],
    mode='markers', name='3 Nearest Neighbors',
    marker=dict(size=12, color='rgba(0,0,0,0)', line=dict(width=3, color='black')) # Black outlines
))

# Add the new point (Blue diamond shape to stand out clearly)
fig.add_trace(go.Scatter3d(
    x=new_setting['Distance'], y=new_setting['Flow-rate'], z=new_setting['Voltage'],
    mode='markers', name=f'New Setting (Predicted: Class {predicted_class})',
    marker=dict(size=14, symbol='diamond', color='blue', line=dict(width=2, color='black'), opacity=1)
))

# 5. Figure layout configuration
fig.update_layout(
    title=dict(text="Normalized KNN Model with 3 Nearest Neighbors", font=dict(size=18), x=0.5),
    scene=dict(xaxis_title='Distance (mm)', yaxis_title='Flow-rate (mL/h)', zaxis_title='Voltage (kV)', bgcolor='white'),
    margin=dict(l=0, r=0, b=0, t=60),
    legend=dict(title="Legend", x=0.85, y=0.9, bordercolor="black", borderwidth=1)
)

# Sauvegarde le graphique dans un fichier interactif autonome
fig.write_html("Graphique_KNN_3D.html")
fig.show()
print("[SUCCÈS] Le fichier Graphique_KNN_3D.html a été créé dans ton dossier !")
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Rectangle
from tqdm.notebook import tqdm
from matplotlib import ticker
import seaborn as sns
# import the data

# link to teams overall stats, has conference only and overall stats
# https://pacathletics.org/teamstats.aspx?path=baseball&year=2023&school=cha&conf=true


# sheet = pd.read_csv("/Users/akulpatel/Desktop/practice/test_practice/Chatham_Baseball_Data.xlsx")
xls = pd.ExcelFile("/Users/akulpatel/Desktop/practice/test_practice/clustersheet.xlsx")
Thornton = pd.read_excel(xls, sheet_name="Thornton", header=None)
McSorely = pd.read_excel(xls, sheet_name="McSorely", header=None)
Phillips = pd.read_excel(xls, sheet_name="Phillips", header=None)
Schwartz = pd.read_excel(xls, sheet_name="Schwartz", header=None)
Cote = pd.read_excel(xls, sheet_name="Cote", header=None)

Thornton.columns = ['Player Name: Thornton Mitchell', 'No', 'Date', 'Pitch ID', 'Pitch Type', 'Is Strike',
                    'StrikeZoneSide', 'StrikeZoneHeight', 'Velocity',
                    'Total Spin', 'True Spin', 'Spin Efficiency', 'Spin Direction', 'Spin Confidence', 'VB', 'HB',
                    'SSW VB', 'SSW HB', 'VB Spin', 'HB Spin'
    , 'Horizontal Angle', 'Release Angle', 'Release Height', 'Release Side', 'RE', 'Gyro', 'UID', 'Serial']

McSorely.columns = ['Player Name: Thornton Mitchell', 'No', 'Date', 'Pitch ID', 'Pitch Type', 'Is Strike',
                    'StrikeZoneSide', 'StrikeZoneHeight', 'Velocity',
                    'Total Spin', 'True Spin', 'Spin Efficiency', 'Spin Direction', 'Spin Confidence', 'VB', 'HB',
                    'SSW VB', 'SSW HB', 'VB Spin', 'HB Spin'
    , 'Horizontal Angle', 'Release Angle', 'Release Height', 'Release Side', 'RE', 'Gyro', 'UID', 'Serial']

Phillips.columns = ['Player Name: Thornton Mitchell', 'No', 'Date', 'Pitch ID', 'Pitch Type', 'Is Strike',
                    'StrikeZoneSide', 'StrikeZoneHeight', 'Velocity',
                    'Total Spin', 'True Spin', 'Spin Efficiency', 'Spin Direction', 'Spin Confidence', 'VB', 'HB',
                    'SSW VB', 'SSW HB', 'VB Spin', 'HB Spin'
    , 'Horizontal Angle', 'Release Angle', 'Release Height', 'Release Side', 'RE', 'Gyro', 'UID', 'Serial']

Schwartz.columns = ['Player Name: Thornton Mitchell', 'No', 'Date', 'Pitch ID', 'Pitch Type', 'Is Strike',
                    'StrikeZoneSide', 'StrikeZoneHeight', 'Velocity',
                    'Total Spin', 'True Spin', 'Spin Efficiency', 'Spin Direction', 'Spin Confidence', 'VB', 'HB',
                    'SSW VB', 'SSW HB', 'VB Spin', 'HB Spin'
    , 'Horizontal Angle', 'Release Angle', 'Release Height', 'Release Side', 'RE', 'Gyro', 'UID', 'Serial']

Cote.columns = ['Player Name: Thornton Mitchell', 'No', 'Date', 'Pitch ID', 'Pitch Type', 'Is Strike', 'StrikeZoneSide',
                'StrikeZoneHeight', 'Velocity',
                'Total Spin', 'True Spin', 'Spin Efficiency', 'Spin Direction', 'Spin Confidence', 'VB', 'HB', 'SSW VB',
                'SSW HB', 'VB Spin', 'HB Spin'
    , 'Horizontal Angle', 'Release Angle', 'Release Height', 'Release Side', 'RE', 'Gyro', 'UID', 'Serial']

selected_columns = ['Pitch Type', 'Is Strike', 'StrikeZoneSide', 'StrikeZoneHeight', 'Velocity',
                    'Total Spin', 'True Spin', 'Spin Efficiency']

players = [Thornton, McSorely, Phillips, Schwartz, Cote]
names = ['Thornton', 'McSorely', 'Phillips', 'Schwartz', 'Cote']
count = 0

for i in players:
    selected_data = i[selected_columns]

    X = np.array(selected_data['StrikeZoneSide'][1:])
    Y = np.array(selected_data['StrikeZoneHeight'][1:])

    data = np.column_stack((X, Y))

    # compute euclidean distance between points
    affinity_matrix = pairwise_distances(data, metric='euclidean')

    # Apply RBF kernel
    gamma = 1
    affinity_matrix = np.exp(-gamma * affinity_matrix ** 2)

    # Spectral Clustering
    num_clusters = 2
    spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
    labels = spectral_clustering.fit_predict(affinity_matrix)

    good_pitches = data[(X >= -10.0) & (X <= 10.0) & (Y >= 15.0) & (Y <= 35.0)]
    # medium_pitches = data[((X >= -15.0) & (X < -10.0) or (X > 10.0) & (X <= 15.0)) & (((Y >= 10.0) & (Y < 15.0)) or ((Y > 35.0) & (Y <= 45.0)))]
    # bad_pitches = data[((X < -15) & (X > 15) & (Y > 45) & (Y < 10))]

    #bad pitch locations
    bad_condition_x = (X < -15) | (X > 15)
    bad_condition_y = (Y < 10) | (Y > 45)

    # Filter out bad pitches
    bad_pitches = data[bad_condition_x | bad_condition_y]

    #average pitch location
    average_condition_x = ((X >= -15) & (X <= -10)) | ((X >= 10) & (X <= 15))
    average_condition_y = ((Y >= 10) & (Y <= 15)) | ((Y >= 35) & (Y <= 45))

    # Filter out average pitches
    average_pitches = data[average_condition_x & average_condition_y]


    # Original grid size
    x_range = [-30, 30]
    y_range = [-10, 60]

    # Plot all pitches
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='yellow')
    grid = plt.scatter(X, Y, color='yellow')
    # plt.scatter(medium_pitches[:0])

    # Plot good pitches in green
    plt.scatter(good_pitches[:, 0], good_pitches[:, 1], color='green', label='Good Pitches')

    # Plot bad pitches in red
    plt.scatter(bad_pitches[:, 0], bad_pitches[:, 1], color='red', label='Bad Pitches')

    # plot the averages pitches in gray
    plt.scatter(average_pitches[:, 0], average_pitches[:, 1], color='yellow', label='Average Pitches')

    # Plot the perimeter
    plt.plot([-10, 10, 10, -10, -10], [15, 15, 40, 40, 15], color='blue', linestyle='--', linewidth=2,
             label='Perimeter')

    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.xlabel('X (Side)')
    plt.ylabel('Y (Height)')
    plt.title(names[count])
    count = count + 1
    plt.legend()
    plt.grid(True)
    plt.show()

selected_data = Thornton[selected_columns]

pitch_type_counts = selected_data['Pitch Type'][1:].value_counts()
pitch_type_counts.plot(kind='bar')
plt.xlabel('Pitch Type')
plt.ylabel('Count')
plt.title('Thornton Pitch Types')
plt.xticks(rotation=45)
plt.show()

#
# plt.hist(selected_data['Velocity'][19:], bins=20)
# plt.xlabel('Velocity')
# plt.ylabel('Frequency')
# plt.title('Thornton Velocity Distribution')
# plt.show()

pitch_type_counts = selected_data['Pitch Type'].value_counts()
plt.pie(pitch_type_counts, labels=pitch_type_counts.index, autopct='%1.1f%%', startangle=45)
plt.axis('equal')
plt.title('Thornton Pitch Type Distribution')
plt.show()


selected_features = ['Velocity', 'Spin Efficiency', 'Total Spin', 'True Spin']
correlation_matrix = selected_data[selected_features][24:].corr()

# correlation matrix heat map for an individual player features.
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Thornton Correlation Matrix of Selected Features')
plt.show()

plt.figure(figsize=(10, 10))
plt.boxplot([selected_data[selected_data['Pitch Type'] == pitch]['Velocity'][24:] for pitch in
             selected_data['Pitch Type'].unique()])
plt.xlabel('Pitch Type')
plt.ylabel('Velocity')
plt.title('Thornton Velocity by Pitch Type')
plt.xticks(range(1, len(selected_data['Pitch Type'].unique()) + 1), selected_data['Pitch Type'].unique(), rotation=45)
plt.show()

# CLEAN DATA
tqdm.pandas()
# Load in sheet and drop all empty cells. Then drop all rows that contained empty cells.
clustersheet_df = pd.read_excel('clustersheet.xlsx')
clustersheet_df.replace('-', pd.NA, inplace=True)
clustersheet_cleaned = clustersheet_df.dropna(
    subset=['Velocity', 'Total Spin', 'Strike Zone Side', 'Strike Zone Height', 'Pitch Type', 'Is Strike'])


feature_analysis = clustersheet_cleaned[
    ['Velocity', 'Total Spin', 'Strike Zone Side', 'Strike Zone Height', 'Pitch Type', 'Is Strike']].describe()

# print outputs.
print('Cleaned Data Head:')
print(clustersheet_cleaned.head())
print('\nFeature Analysis for Pitching Statistics:')
print(feature_analysis)

sns.set_style('whitegrid')

# get player averages
player_means = clustersheet_cleaned.groupby('Player Name:')[
    ['Velocity', 'Total Spin', 'Strike Zone Side', 'Strike Zone Height']].mean()
player_means.reset_index(inplace=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot each of the features
for i, feature in enumerate(['Velocity', 'Total Spin', 'Strike Zone Side', 'Strike Zone Height']):
    ax = axes[i // 2, i % 2]
    sns.barplot(x=feature, y='Player Name:', data=player_means.sort_values(feature, ascending=False), ax=ax)
    ax.set_title('Average ' + feature + ' by Player')

plt.tight_layout()
plt.show()

sns.set_style('whitegrid')
fig, axes = plt.subplots(4, 1, figsize=(10, 20))

#plot averages
sns.barplot(x='Velocity', y='Player Name:', data=player_means.sort_values('Velocity', ascending=False), ax=axes[0],
            palette='viridis')
axes[0].set_title('Average Velocity by Player')
# total spin
sns.barplot(x='Total Spin', y='Player Name:', data=player_means.sort_values('Total Spin', ascending=False), ax=axes[1],
            palette='magma')
axes[1].set_title('Average Total Spin by Player')
# average strike zone side
sns.barplot(x='Strike Zone Side', y='Player Name:', data=player_means.sort_values('Strike Zone Side', ascending=True),
            ax=axes[2], palette='cubehelix')
axes[2].set_title('Average Strike Zone Side by Player')
# average strike zone height
sns.barplot(x='Strike Zone Height', y='Player Name:',
            data=player_means.sort_values('Strike Zone Height', ascending=False), ax=axes[3], palette='coolwarm')
axes[3].set_title('Average Strike Zone Height by Player')

plt.tight_layout()
plt.show()

# Analyze and rank players off pitch type
pitch_type_strikes = clustersheet_cleaned.groupby(['Player Name:', 'Pitch Type'])['Is Strike'].apply(
    lambda x: (x == 'Y').sum()).reset_index(name='Strike Count')
pitch_type_strikes['Rank'] = pitch_type_strikes.groupby('Pitch Type')['Strike Count'].rank(method='max',
                                                                                           ascending=False)
pitch_type_strikes_sorted = pitch_type_strikes.sort_values(['Pitch Type', 'Rank'])

plt.figure(figsize=(15, 10))
sns.barplot(x='Strike Count', y='Player Name:', hue='Pitch Type', data=pitch_type_strikes_sorted, palette='Set2')
plt.title('Players Ranked by Most Strikes with Pitch Type')
plt.xlabel('Number of Strikes')
plt.ylabel('Player Name')
plt.legend(title='Pitch Type')
plt.tight_layout()
plt.show()


player_averages = clustersheet_cleaned.groupby('Player Name:')[['Velocity', 'True Spin (release)', 'Spin Efficiency (release)']].mean().reset_index()


def create_precise_chart(data, y, title, tick_spacing=10):
    plt.figure(figsize=(14, 8))
    chart = sns.barplot(x='Player Name:', y=y, data=data.sort_values(by=y, ascending=False))
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    chart.set_title(title)
    chart.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    chart.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()
    plt.show()

# Create precise charts for player averages
create_precise_chart(player_averages, 'Velocity', 'Average Velocity by Player', 5)
create_precise_chart(player_averages, 'True Spin (release)', 'Average True Spin by Player', 100)
create_precise_chart(player_averages, 'Spin Efficiency (release)', 'Average Spin Efficiency by Player', 5)


pitch_type_encoded = pd.get_dummies(clustersheet_cleaned['Pitch Type'], prefix='Pitch_Type')
is_strike_encoded = pd.get_dummies(clustersheet_cleaned['Is Strike'], prefix='Is_Strike', drop_first=True)

encoded_df = pd.concat([clustersheet_cleaned[['Strike Zone Side', 'Strike Zone Height', 'Velocity', 'Total Spin', 'True Spin (release)', 'Spin Efficiency (release)']], pitch_type_encoded, is_strike_encoded], axis=1)

#overall correlation matrix for every features.
correlation_matrix = encoded_df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


#CLUSTERING ANALYSIS

# Load cleaned data
clustersheet = pd.read_excel('clustersheet_cleaned.xlsx')

# strike zone boundaries.
strike_zone = {'left': -10, 'right': 10, 'bottom': 18, 'top': 45}

#perform a cluster analysis for each players strike zone data
for player_name, group in tqdm(clustersheet.groupby('Player Name:')):
    #sort by non-duplicate pitches and pitch types
    unique_pitch_types = group['Pitch Type'].nunique()
    pitch_types = group['Pitch Type'].dropna().unique()

    #spectral clustering
    spectral = SpectralClustering(n_clusters=unique_pitch_types, affinity='nearest_neighbors')
    features = group[['Strike Zone Side', 'Strike Zone Height']].dropna().values
    labels = spectral.fit_predict(features)

    # Plot each cluster
    plt.figure(figsize=(10, 8))
    for label, pitch_type in zip(np.unique(labels), pitch_types):
        cluster_data = features[labels == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=pitch_type, s=50)
        #draw a region around the most concentrated areas
        nbrs = NearestNeighbors(n_neighbors=2).fit(cluster_data)
        distances, indices = nbrs.kneighbors(cluster_data)
        avg_distance = np.mean(distances)
        density_threshold = avg_distance / 2
        dense_areas = cluster_data[np.mean(distances, axis=1) < density_threshold]
        if dense_areas.size > 0:
            x_min, y_min = np.min(dense_areas, axis=0)
            x_max, y_max = np.max(dense_areas, axis=0)
            plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='red', linewidth=0.5))

    # Draw the strike zone
    plt.gca().add_patch(Rectangle((strike_zone['left'], strike_zone['bottom']),
                                  strike_zone['right'] - strike_zone['left'],
                                  strike_zone['top'] - strike_zone['bottom'],
                                  fill=False, edgecolor='black', linewidth=2, linestyle='-'))
    plt.title(f'Spectral Clustering for {player_name}')
    plt.xlabel('Strike Zone Side')
    plt.ylabel('Strike Zone Height')
    plt.legend()
    plt.grid(True)

    # Save each plot as png's
    filename = f'spectral_clustering_{player_name}.png'
    plt.savefig(filename)
    plt.close()
    print(f'Plot saved for : {player_name}')


def cluster_and_plot(player_group):
    player, group = player_group
    # unique non-duplicated pitches
    pitch_types = group['Pitch Type'].unique()
    n_clusters = len(pitch_types)
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    features = group[['Strike Zone Side', 'Strike Zone Height']].dropna()
    kmeans.fit(features)
    # Predict the concentrated clusters
    features['Cluster'] = kmeans.predict(features)
    # lables
    cluster_pitch_map = {i: pitch_types[i] for i in range(n_clusters)}
    features['Pitch Type Label'] = features['Cluster'].map(cluster_pitch_map)
    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Strike Zone Side', y='Strike Zone Height', data=features, hue='Pitch Type Label',
                    palette='viridis', legend='full')
    plt.title('Unique Pitch Type Clustering for ' + player)
    plt.xlabel('Strike Zone Side')
    plt.ylabel('Strike Zone Height')
    plt.legend(title='Pitch Type')
    plt.show()


# run function
for player_group in tqdm(clustersheet_cleaned.groupby('Player Name:')):
    cluster_and_plot(player_group)

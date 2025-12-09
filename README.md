# CourtDynamics-GroupGhostbusters-315881
Court Dynamics: Unsupervised Discovery of NBA Player Roles

Team Members

Taha Berk Altin (Captain, 315881)

Fatma Sena Yildiz (323281)

Asmin Balku (321211)

1. Introduction

In the modern NBA, the concept of "position" has become increasingly fluid. Traditional labels like Point Guard (PG) or Center (C) often fail to capture the functional role of a player on the court. For instance, a "Point Forward" like LeBron James operates as a primary playmaker despite being the size of a Power Forward, while a "Stretch Five" operates on the perimeter despite being a Center.
The goal of the Court Dynamics project is to move beyond these biological or historical labels. By applying Unsupervised Machine Learning to the basketball.db dataset, we aim to discover latent player archetypes based solely on statistical performance. This project specifically focuses on:
Clustering: Grouping players into performance tiers (e.g., "High-Usage Scorers" vs. "Defensive Specialists").
Outlier Detection: Identifying unique players who defy standard categorization.
League Structure: Analyzing how players from different eras (NBA vs. ABA) mathematically separate in the data.

2. Methods: Data & Preprocessing

2.1 The Dataset

We utilized the basketball.db SQLite database, specifically determining that the player_regular_season_career table offered the most robust aggregate view of a player's historical impact.
Total Records: 3,784 unique player profiles.
Feature Selection: We extracted 18 numerical features covering three distinct facets of the game:
Offensive Production: pts (Points), asts (Assists), fgm (Field Goals Made), tpm (3-Pointers Made).
Defensive Metrics: reb (Rebounds), stl (Steals), blk (Blocks).
Durability/Volume: minutes(Total Minutes Played), gp (Games Played), turnover.

2.2 Data Cleaning & PreprocessingOur Exploratory Data Analysis (EDA) revealed significant inconsistencies due to historical tracking changes. To address this, we implemented a strict preprocessing pipeline:

Handling Missing Values (Imputation):
Problem: Stats like tpa (3-Point Attempts), tpm, stl(Steals), and blk(Blocks) were not tracked in the early NBA/ABA years, leading to NaN values for older players.
Solution: We applied Median Imputation using SimpleImputer. We rejected Mean Imputation because NBA statistics are heavily right-skewed (superstars have massive outliers), meaning the mean would artificially inflate the stats for bench players, providing an inaccurate baseline for role players.The median provided a robust central tendency that is resistant to these outliers.

Categorical Encoding:

The dataset contained a leag variable (League) indicating if a player was in the NBA or ABA.
We applied One-Hot Encoding to create leag_N and leag_A. This allowed our distance-based algorithms to mathematically differentiate between the two leagues. Instead of assigning arbitrary numeric labels (e.g., NBA = 0, ABA = 1) that could introduce a false ordinal relationship (e.g ABA>NBA):, One-Hot Encoding represents each league as a vector such as [1, 0] or [0, 1], ensuring that no artificial ordering is introduced. This encoding preserves the categorical nature of the feature while making it compatible with Euclidean-based models such as K-Means and DBSCAN.

Feature Scaling: This is the most critical step for our distance based algorithms (K-means,DBSCAN, and  also improves the stability of our anomally detector (Isolation Forest))
We applied StandardScaler to transform all features to a mean of 0 and variance of 1.
Justification: Without scaling, volume metrics like minutes (ranging up to 50,000) would completely dominate the Euclidean distance calculations, rendering smaller but crucial metrics like stl (ranging 0–2,000) irrelevant Another example :high-magnitude features like minutes played (up to 50,000) would dominate distance calculations, making smaller but meaningful statistics such as steals essentially invisible to the algorithm.
By standardizing, you ensure that a player who is an outlier in Blocks (e.g., Manute Bol) pulls the algorithm's attention just as much as a player who is an outlier in Minutes (e.g., Wilt Chamberlain). This produces more balanced clusters and more meaningful anomaly detection results.

3. Experimental Design:

3.1. Problem Definition and Our Approach

Since the NBA player data in the "Court Dynamics" project does not contain any pre-existing labels (such as "Good Player" or "Bad Player"), we structured this study as an Unsupervised Learning problem. Our central focus was on detecting the implicit structural dynamics and undefined player roles embedded in the data.
To capture different characteristics of the dataset, we did not rely on a single model but instead tested three distinct clustering approaches:
K-Means: To divide players into distinct, non-overlapping performance tiers.
DBSCAN: To detect the league's "exceptional" players who do not fit standard patterns.
Hierarchical Clustering: To observe the relationships and sub-branches between player groups.
Since we did not have "Ground Truth" labels to verify our results, we validated our models using mathematical metrics like the Silhouette Score and the Elbow Method. Additionally, to visualize our complex 18-dimensional dataset with the human eye, we used PCA (Principal Component Analysis) to reduce the dimensions to two.

3.2. Model Validation and Experimental Process

The most critical decision in clustering algorithms is determining how many parts to divide the data into (the number K). Instead of picking a random number, we listened to the mathematics of the data and employed two fundamental methods.

3.2.1. Selection of Optimal K with the Elbow Method

<img width="736" height="484" alt="image" src="https://github.com/user-attachments/assets/0d76fed5-a926-4664-8ad0-382a2b4d2db3" />


For the K-Means algorithm, we plotted the error rate against the number of clusters. We were looking for that specific "breaking point" where the speed of error reduction suddenly slows down.
Our Analysis: Upon examining the graph, we observed that the error dropped drastically from K=2 to K=4 (falling from levels of 46,000 down to 32,000), but after 4, this drop became marginal.
Our Decision: Adding a 5th or 6th cluster to the system did not significantly increase the model's success; on the contrary, it unnecessarily increased complexity. Therefore, we decided on K=4 as the mathematical saturation point.

3.2.2. Verification with Silhouette Analysis

<img width="718" height="484" alt="image" src="https://github.com/user-attachments/assets/ae436a7e-5843-4f2e-b4cf-c76a83a11ae2" />


Instead of relying solely on the Elbow method, we also calculated the Silhouette Score, which measures the quality of the clusters.
Why not K=2? Although K=2 gave the highest score mathematically (0.61), simply splitting the data into "Good Players" and "Bad Players" did not serve the purpose of our project (finding nuanced player archetypes).
Why K=4? The value of K=4 produced a higher score (0.5721) compared to K=3, creating a local peak. This proved to us that K=4 offers a level of detail that is both statistically consistent and meaningful for basketball analytics.

3.3. Algorithms Implemented and Findings

3.3.1. Algorithm 1: K-Means

We chose K-Means as the backbone of the project. This algorithm grouped players based on their distance to the 4 centroid points we determined.
Settings: We used the k-means method to prevent the initial centers from being selected randomly and giving incorrect results.
Result: The model successfully separated bench players, superstars, rotation players, and interestingly, the league distinction (ABA vs. NBA) very clearly. This was the most successful model for assigning players to specific "roles."

3.3.2. Algorithm 2: DBSCAN

While K-Means tries to force every player into a group, we used the DBSCAN algorithm to find those who "stray from the herd." Since this algorithm works based on density, it marks points that do not fit anywhere as "Noise" (-1).
Settings: We selected the parameters eps=2 and min_samples=5.
Findings:
Cluster -1 (Blue Points): The model isolated the legendary players who are far above the league average and marginal players who played very few games as "Noise."
How We Interpreted This: This is not an error, but rather a success. Because names like Michael Jordan or Wilt Chamberlain really cannot be fitted into an "average" cluster. DBSCAN allowed us to mathematically detect these exceptions (outliers).

3.3.3. Algorithm 3: Hierarchical Clustering

Finally, we used Hierarchical Clustering (Agglomerative) to map out the family tree of the dataset. This method builds a tree structure by merging players based on their similarities.
Parameters:
linkage = 'ward': Chosen to minimize variance increase when merging clusters.
affinity = 'euclidean': Distance metric used.

Dendrogram Analysis: The generated dendrogram reveals a major primary split in the natural structure of the data. This visualization confirms that the 4 clusters found by K-Means are not random, but that these divisions (NBA vs. ABA and High vs. Low Performance) inherently exist in the root structure of the data.

<img width="1255" height="666" alt="image" src="https://github.com/user-attachments/assets/c7b26f95-1e1b-431c-9188-62f7fcf04fd7" />


3.4. Visualization (PCA)
We had 18 different features like points, rebounds, assists, etc. Since it is impossible to imagine an 18-dimensional world, we used PCA (Principal Component Analysis) for reporting.
We can read the axes in our graphs as follows:
Horizontal Axis (PCA 1): Represents the player's volume (Usage rate, Minutes played). As you move to the right, the player's "stardom" increases.
Vertical Axis (PCA 2): Represents the playstyle (Bigs vs. Guards)

3.5. Outcomes
As a result of our experiments, we reached the following decision:
K-Means: K-Means gave the most successful and understandable results in separating players into clear roles like "Star," "Starter," and "Bench." The choice of K=4 was also statistically verified.
DBSCAN: Although it remained too rigid for a general classification, it was much more successful than K-Means in detecting exceptional players (outliers).
Hierarchical Clustering: Served as a verification tool to confirm the relationships between clusters.
Final Decision: We based our player profile analysis on the K-Means (K=4) model, but we utilized DBSCAN results to interpret outlier values.

4. Results and Main Findings

4.1 Optimal Clustering Configuration (K-Means)

At the end of our analysis we identified K=4 as the optimal number of clusters.
Elbow Analysis: The inertia plot showed a distinct "bend" at K=4, where the drop in error slowed significantly. 
Silhouette Score: K=4 achieved a strong score of 0.5721. While K=2 was mathematically higher, it provided insufficient assessment (by splitting only "Good" vs. "Bad"). K=4 offered the best balance of stability and interpretability.
After having done these different methods, we came to the conclusion that the Elbow Method was better because it revealed a 4-cluster structure that captured critical points like the ABA league, whereas the Silhouette Method recommended a simple 2-group split that missed these meaningful distinctions.

4.2 Cluster Interpretation (K-Means Analysis)

<img width="859" height="561" alt="image" src="https://github.com/user-attachments/assets/1eb01743-ae26-41b2-8f77-bab15592de8d" />

We grouped players into four distinct types with the K-Means algorithm by using 4 clusters (K=4). By looking at the average player (centroid) in each group, we identified these specific roles: 

Cluster 0: The "Rotation & Role" Group (Low Numbers)
Stats: This group has the lowest averages for minutes, points, and games played.
Meaning: This represents the majority of professional players. These are players with short careers, bench players, or specialists who don't play many minutes. Their impact is average or "replacement level."
Finding: Since this is the largest group, we were able to confirm that having a long, high-scoring career in the NBA is very rare. Most players do not stay in high-volume roles for a long time.

Cluster 1: The "Superstars" 
Stats: This group contains outliers with massive numbers in minutes, points, assists, and rebounds.
Meaning: These are the kind of players we can call “All-Time Legends”. The algorithm grouped them together not just because they scored a lot, but because they played for a long time (durability). They are the only players with stats high enough to change the shape of the entire dataset.
Type: "The 1st Option." These players define their era.

Cluster 2: The "ABA Legacy" Group
Stats: The main feature of this group is that they have a value of "1.0" for the ABA league variable.
Meaning: This is a structural group. The model successfully realized that players from the American Basketball Association (1967–1976) were statistically different. They played in a different environment with a faster pace and an earlier 3-point line than the NBA.
Proof: This proves the model pays attention to context. So it’s separated players by where they played, not just how they played.

Cluster 3: The "Core Starters" (High Efficiency)
Stats: These players have high games played and good efficiency, but their usage is significantly lower than Cluster 1.
Meaning:  These are consistent starting players who are essential to their teams. They have long careers and play many minutes, but they tend to specialize in specific areas (such as scoring or defense) rather than dominating every statistical category like the players in Cluster 1.

4.3 Outlier Detection (DBSCAN Findings)

<img width="859" height="561" alt="image" src="https://github.com/user-attachments/assets/aeef7584-0095-4a06-a701-119245f70fa9" />

While K-Means forced every player into a group, the DBSCAN algorithm allowed us to find players who did not fit into any group. 
The "Noise" Points (Cluster -1):
Observation: The algorithm labeled a small number of players as "-1". So we can identify them as "noise" or errors (meaning they are superstars).
Analysis: These players (e.g. legends like Michael Jordan or LeBron James) have stats that are so unique they don't have any "neighbors" in the data. They stand alone.
Conclusion: This confirms that "Superstar" is a mathematical anomaly here.  A normal evaluation model might fail to predict their value because they exist outside the normal range of data.

5. Conclusion

5.1 Conclusions: The "Court Dynamics" Framework

Take-Away Point:

Our work proves that a player's "biological" position (like Guard or Center) is less important than their Volume and Era. The data showed that the mathematical difference between a "Bench Guard" and a "Star Guard" is much bigger than the difference between a "Star Guard" and a "Star Center." Unsupervised learning grouped players primarily by their Impact Tier (how much they contributed) rather than their playing style. This means that for building a team, "Talent Level" is a stronger defining feature than "Position." Additionally, the model proved that historical context is critical by automatically separating players from the ABA league, showing that stats from different eras cannot be treated as the same


5.2 Projecting Trajectories

Our current model looks at a player's total career averages. To test our ideas further, we created a simple chart  that compares the performance profiles of the top 3 scorers in history.

Current Prototype: We grouped the data by player and calculated the average for important stats like games played, minutes, points, and rebounds.

Observation: The chart shows that "Superstars" (Cluster 1) have a specific shape. They have very high numbers for minutes and points, but they differ in other stats like assists or rebounds.

Limitation: This method combines a 20-year career into one single average number. This hides important details, like when a player started to decline or when they had their best season.

The "Growth Trajectories" (Next Step): 

One question not fully answered by our work is when a player transitions from one role to another. Our current model uses career totals, which flattens a player's entire development into a single static number. It does not show the difference between a player who was always a "Role Player" and one who started as a "Star" but declined. A natural next step for this research is to implement Time-Series Clustering. By tracking a player's stats season-by-season, we could build a predictive model to forecast the probability of a rookie developing into a "Cluster 1" superstar, providing valuable insight for scouts.

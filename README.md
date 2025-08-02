🛍️ Customer Segmentation Using Wholesale Spending Patterns
This project explores the Wholesale Customers Dataset from UCI, which includes annual spending data across six product categories. The goal is to apply K-Means clustering to segment customers into meaningful groups based on their purchasing behaviour.
🎯 Project Objective
Use unsupervised machine learning to identify customer segments, enabling businesses to:
•	Understand distinct customer groups
•	Target promotions more effectively
•	Personalize product offerings
•	Improve marketing strategy and inventory planning
________________________________________
📦 Tools & Technologies
•	Python
•	Pandas, NumPy
•	scikit-learn for preprocessing and clustering
•	Matplotlib, Seaborn for data visualization
•	Streamlit (for deployment, if used)
________________________________________
📊 Key Steps & Analysis
🔍 Exploratory Data Analysis (EDA)
•	Pair plots and heatmaps revealed strong correlations (e.g., between Grocery, Milk, and Detergents_Paper)
•	Count plots showed the distribution of customers by Region and Channel
•	Correlation bar charts highlighted the strongest feature relationships
🔄 Data Preprocessing
•	Feature scaling using StandardScaler
•	Selection of optimal clusters using the Elbow Method
•	Cluster labeling: High Spenders, Mid-Range Buyers, and Budget Shoppers
🔐 Clustering
•	KMeans(n_clusters=3) was selected as optimal
•	New segment labels were mapped based on cluster behavior
•	Visualizations with pairplots and scatterplots helped interpret cluster patterns
________________________________________
📈 Insights
•	Segment 0 – Budget Shoppers: Lowest spending across categories
•	Segment 1 – Mid-Range Buyers: Moderate, balanced spending
•	Segment 2 – High Spenders: Heavy spending on Grocery, Milk, and Detergents
________________________________________
🚀 How to Run the Project
pip install -r requirements.txt
python customer_segmentation.py
If using Streamlit:
streamlit run customer_segmentation.py
________________________________________
🧾 Requirements
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
________________________________________
📌 Conclusion
This project demonstrates how unsupervised learning techniques like K-Means can uncover patterns in customer spending, helping businesses make data-driven decisions.

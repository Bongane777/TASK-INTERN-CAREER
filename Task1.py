#!/usr/bin/env python
# coding: utf-8

# DATA ANALYSIS ON YOUTUBERS

# In[3]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 


# In[4]:


df = pd.read_csv("youtubers_df.csv")


# In[6]:


# Display basic information about the dataset
print("Dataset Information:")
print(df.info())


# In[8]:


# Display summary statistics for numeric variables
print("\nData Statistics:")
print(df.describe())


# In[9]:


# Display the first few rows of the dataset
print("\nFirst 10 Rows of the Dataset:")
print(df.head(10))


# DATA CLEANING

# In[19]:


# Check for missing data
missing_data = df.isnull()

# Sum the missing values for each column
missing_counts = missing_data.sum()

# Display the number of missing values for each column
print("Missing Data Summary:")
print(missing_counts)

# Handling missing values
df.dropna(subset=['Categories'], inplace=True)  # Drop rows with missing 'Categories'
df['Country'].fillna('Unknown', inplace=True)  # Fill missing 'Country' with 'Unknown'

# Display the cleaned dataset
print("\nCleaned Dataset:")
print(df)



# 

# CHECK FOR OUTLIERS AND REMOVE OUTLIERS

# In[22]:


# Box plot for 'Subscribers'
sns.boxplot(data=df['Suscribers'])
plt.title('Box Plot of Subscribers')
plt.show()


# In[25]:


# Box plot for 'Visits'
sns.boxplot(data=df['Visits'])
plt.title('Box Plot of Visits')
plt.show()


# In[26]:


# Box plot for 'Likes'
sns.boxplot(data=df['Likes'])
plt.title('Box Plot of Likes')
plt.show()


# In[27]:


# Function to remove outliers using IQR method
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_filtered = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data_filtered


# In[28]:


# Remove outliers for 'Subscribers'
df_cleaned_subscribers = remove_outliers(df, 'Suscribers')
# Box plot for 'Subscribers'
sns.boxplot(data=df_cleaned_subscribers)
plt.title('Box Plot of Likes')
plt.show()


# In[29]:


# Remove outliers for 'Visits'
df_cleaned_visits = remove_outliers(df, 'Visits')
# Box plot for 'Visits'
sns.boxplot(data=df_cleaned_visits)
plt.title('Box Plot of Likes')
plt.show()


# In[ ]:


# Remove outliers for 'Likes'
df_cleaned_likes = remove_outliers(df, 'Likes')
# Box plot for 'Likes'
sns.boxplot(data= df_cleaned_likes)
plt.title('Box Plot of Likes')
plt.show()


# TREND ANALYSIS

# In[30]:


# Visualize the distribution of categories using a bar plot
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Categories', order=df['Categories'].value_counts().index, palette='viridis')
plt.title('Distribution of Categories Among Top YouTube Streamers')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.xlabel('Categories')
plt.ylabel('Number of Streamers')
plt.show()


# In[67]:


print("According to the graph above the most popular categories are Musica baile", "Peliculas,Animacion", "Musica y baile,Peliculas",
"Vlogs diarios", "Noticias y Politica.")


# Correlation

# In[33]:


# Calculate correlation coefficients
correlation_likes = df['Suscribers'].corr(df['Likes'])
correlation_comments = df['Suscribers'].corr(df['Comments'])

print(f"Correlation between Subscribers and Likes: {correlation_likes:.2f}")
print(f"Correlation between Subscribers and Comments: {correlation_comments:.2f}")


# In[68]:


print("The correlation between the number of subscribers and the number of likes is 0.25 which is a weak and positive correlation, this means that changes in the number of subscribers are only weakly associated with changes in the number of likes.")


# In[69]:


print("The correlation between Subscribers and comments is 0.04 which is weak and positive, this means that changes in the number of subscribers are weakly associated with changes in the number of comments, and the relationship is almost negligible.")


# In[38]:


# Explore the distribution of streamers' audiences by country
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Country', palette='viridis', order=df['Country'].value_counts().index)
plt.title('Distribution of Streamers\' Audiences by Country')
plt.xlabel('Country')
plt.ylabel('Number of Streamers')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()


# In[42]:


# Analyze regional preferences for specific content categories
regional_preferences = df.groupby('Country')['Categories'].value_counts()
print("\nRegional Preferences for Content Categories:")
print(regional_preferences)


# In[70]:


print("According to the statistics above it shows there isn't much regional preference for specific categories ")


# PERFORMANCE METRICS

# In[72]:


# Calculate average metrics
average_subscribers = df['Suscribers'].mean()
average_visits = df['Visits'].mean()
average_likes = df['Likes'].mean()
average_comments = df['Comments'].mean()

# Visualize the averages
metrics = ['Subscribers', 'Visits', 'Likes', 'Comments']
averages = [average_subscribers, average_visits, average_likes, average_comments]

plt.bar(metrics, averages, color=['blue', 'green', 'orange', 'red'])
plt.title('Average Performance Metrics')
plt.ylabel('Average Count')
plt.show()


# In[73]:


print("According to the graph provided above there are no patterns on the data but there are anomalies as we can see that subscribers are the outliers of the data and comments do not even appear on the graph")


# In[45]:


# Explore the distribution of content categories
category_distribution = df['Categories'].value_counts()

# Display the distribution
print("Distribution of Content Categories:")
print(category_distribution)


# In[74]:


print("Based on the information or statistics givern above the category with exceptional performance is Música y baile which has the highest number of streamers of 160. ")


# BENCHMARKING

# In[60]:


# Calculate average metrics
average_subscribers = df['Suscribers'].mean()
average_visits = df['Visits'].mean()
average_likes = df['Likes'].mean()
average_comments = df['Comments'].mean()

# Filter streamers with above-average performance
above_average_streamers = df[
    (df['Suscribers'] > average_subscribers) &
    (df['Visits'] > average_visits) &
    (df['Likes'] > average_likes) &
    (df['Comments'] > average_comments)
]

# Display the streamers with above-average performance
print("Streamers with Above-Average Performance:")
print(above_average_streamers)


# In[66]:


print("Based on the data above the best content bcreators are Mr Beast and PewDiePie ")


# Content Recommendations

# In[65]:


print("Based on performance metrivcs and streamers category youtubers can enhance content creation by applying these ideas:")
print("Improve the way that YouTube streamers' content is categorized by applying natural language processing (NLP) methods or sophisticated machine learning models. This will guarantee a more precise comprehension of the content categories to which every streamer is assigned.")
print("Integration of Performance Metrics:")
print("Incorporate the performance metrics of streamers (such likes, comments, visits, and subscriptions) into the recommendation system. This will make recommendations that are in line with user tastes by helping to discover popular and trending broadcasters.")
print("Information about User Interaction:")
print("Make advantage of user interaction data to comprehend individual user preferences, such as viewing history, likes, and dislikes. Examine this data to find trends and suggest content based on users' past interactions.")


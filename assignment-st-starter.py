# import packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# show the title
st.title('Data Visualization - Titanic Survival Analysis by Yutong Du')

# read csv and show the dataframe
df = pd.read_csv('train.csv')
st.subheader('Titanic Dataset')
st.write(df.head())
# create a figure with three subplots, size should be (15, 5)
# show the box plot for ticket price with different classes
# you need to set the x labels and y labels
# a sample diagram is shown below
st.subheader('Box Plot for Ticket Prices by Passenger Class')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(y='Fare', data=df[df['Pclass'] == 1], ax=axes[0])
axes[0].set_title('Class 1 Ticket Prices', pad=20)
axes[0].set_xlabel('Class 1')
axes[0].set_ylabel('Fare')

sns.boxplot(y='Fare', data=df[df['Pclass'] == 2], ax=axes[1])
axes[1].set_title('Class 2 Ticket Prices', pad=20)
axes[1].set_xlabel('Class 2')
axes[1].set_ylabel('Fare')

sns.boxplot(y='Fare', data=df[df['Pclass'] == 3], ax=axes[2])
axes[2].set_title('Class 3 Ticket Prices', pad=20)
axes[2].set_xlabel('Class 3')
axes[2].set_ylabel('Fare')

st.pyplot(fig)

st.subheader('Survival Rate by Passenger Class and Survival Status')

grouped = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)

totals = grouped.sum(axis=1)

percentages = (grouped.T / totals).T * 100

values = percentages.values.flatten()

x_labels = [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(values)), values)

ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels)

ax.set_xlabel('(Pclass, Survived)')
ax.set_ylabel('Percentage (%)')
ax.set_title('Survival Rate by Ticket Class and Survival Status')

st.pyplot(fig)

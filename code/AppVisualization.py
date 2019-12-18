import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"E:\yanyi\DataMining\GooglePlayStore\cleanedApp1113.csv")
print(df.info())
df['main_category'] = df['main_category'].astype('category')
df['content_rating'] = df['content_rating'].astype('category')
# df['len_title'] = df['title'].str.len()
# score 分布
print(df['rating'].describe())

g = sns.kdeplot(df.rating, color="Red", shade=True)
g.set_xlabel("rating")
g.set_ylabel("Frequency")
plt.savefig('./rating_distribution.png', dpi=1200)
plt.show()

cate = df['main_category'].value_counts()[0:33]
sns.barplot(x=list(cate.index), y=cate.values)
plt.xticks(rotation=90)
plt.xlabel('category')
plt.ylabel('App quantity')
plt.show()

# 变量相关系数
# corr = df[['rating', 'installs', 'reviews', 'size', 'free', 'price', 'iap', 'iap_min', 'iap_max',
#            'updateDays', 'main_category', 'len_title', 'android_version']].corr()
# print(corr)
# sns.heatmap(abs(corr))
# plt.show()

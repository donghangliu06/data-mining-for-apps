import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from googletrans import Translator
# 1 读入数据
df = pd.read_csv("E:\yanyi\DataMining\GooglePlayStore\AppData1109.csv")
df.info()
pd.set_option('display.max_columns', None)  # 显示所有列
# 2 选择属性
df = df[['title', 'category', 'score', 'histogram', 'description', 'installs', 'reviews',
          'size', 'content_rating', 'free', 'price', 'iap', 'iap_range', 'updated',
          'required_android_version']]
df.info()
print(df.head(5))
print(len(df))
print(len(df.drop_duplicates()))
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)
# 3 改变数据类型
# 3-0 title, description检测语言并翻译为英文
# translator = Translator(service_urls=['translate.google.co.kr',
#                                       'translate.google.com'])
# titletrans = list()
# for tt in deid:
#     print(tt)
#     titletrans.append(translator.translate(df['title'][tt]).text)
# descriptiontrans = list()
# for tt in deid:
#     print(tt)
#     descriptiontrans.append(translator.translate(df['description'][tt][0:400]).text)
detect = pd.read_csv("E:\yanyi\DataMining\GooglePlayStore\detect.txt",
                     names=["deid"])
deid = pd.Series(list(detect['deid']))
entitle = pd.read_csv(r"E:\yanyi\DataMining\GooglePlayStore\en_title.csv",
                      names=["title"])
entitle = entitle.set_index(deid)
endescription = pd.read_csv(r"E:\yanyi\DataMining\GooglePlayStore\en_description.csv",
                            names=["description"])
endescription = endescription.set_index(deid)
df.loc[deid, 'title'] = entitle.loc[deid, 'title']
df.loc[deid, 'description'] = endescription.loc[deid, 'description']
# 3-1 category
df['main_category'] = df['category'].str.split('[,|\'\[\]]', expand=True)[2]
# GAME大类
df.loc[df['main_category'].str[0:4] == 'GAME', 'main_category'] = 'GAME'

# 3-2 histogram
# 去除score缺失行
df = df.dropna(subset=['score'])
df['histogram'] = df['histogram'].str.replace('\s', '')
df[['rating#5', 'rating#4', 'rating#3', 'rating#2', 'rating#1']] \
     = df['histogram'].str.split('[:|,|}]', expand=True)[[1, 3, 5, 7, 9]]
df['rating#5'] = df['rating#5'].astype('int')/df['reviews']
df['rating#4'] = df['rating#4'].astype('int')/df['reviews']
df['rating#3'] = df['rating#3'].astype('int')/df['reviews']
df['rating#2'] = df['rating#2'].astype('int')/df['reviews']
df['rating#1'] = df['rating#1'].astype('int')/df['reviews']
# 3-3 installs整型
df['installs'] = df['installs'].str.replace('+', '')
df['installs'] = df['installs'].str.replace(',', '')
df['installs'] = df['installs'].astype('int64')
# 3-4 size统一单位M
print(df['size'].value_counts())
df['size'] = df['size'].str.replace('M', '')
df['size'] = df['size'].str.replace('k', 'e-3')
df['size'] = df['size'].str.replace('Varies with device', '')
# 3-5 content_rating
df['content_rating'] = df['content_rating'].str.partition(',')[0]
df['content_rating'] = df['content_rating'].str.replace(r'[\'\[\]]', '')
# 3-6 price浮点型
df['price'] = df['price'].str.replace('$', '').astype('float')
# 3-7 iap_range
# 众数填充类似['Digital Purchases']
df.loc[df['iap_range'].str[0] == '[', 'iap_range'] = \
    df['iap_range'].value_counts().index[0]
df.loc[df['iap_range'].str[0] == '(', 'iap_min'] = \
    df.loc[df['iap_range'].str[0]
           == '(', 'iap_range'].str.split('[,|$|\'\(\)]', expand=True)[3]
df.loc[df['iap_range'].str[0] == '(', 'iap_max'] = \
    df.loc[df['iap_range'].str[0]
           == '(', 'iap_range'].str.split('[,|$|\'\(\)]', expand=True)[7]
df[['iap_min', 'iap_max']] = df[['iap_min', 'iap_max']].astype('float')
# 3-8 updated日期型，添加新列updateDays
df['updated'] = pd.to_datetime(df['updated'])
df['updateDays'] = (df['updated'] - df['updated'].max()).dt.days
# 3-9 required_android_version x.x
df['required_android_version'].value_counts()
df['required_android_version'] = df['required_android_version'].str[0:3]
df['required_android_version'] = df['required_android_version'].astype('category')
print(df['required_android_version'].value_counts())
# 3 处理缺失值
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# 变量相关系数
corr = df.drop(columns='rating#3').corr()
print(corr)
sns.heatmap(abs(corr))
plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükle
path = "netflix_titles.csv"
df = pd.read_csv(path)
import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df = df[df["type"] == "Movie"].copy()
    df = df.dropna(subset=["duration", "release_year"])
    df = df[df["duration"].str.contains("min")]
    df["duration"] = df["duration"].str.replace(" min", "").astype(int)
    return df


# 1. Hangi sütunlar eksik?
print("Eksik değer sayıları:")
print(df.isnull().sum())

# 2. Tarih formatı ve dönüşüm
# date_added sütunu datetime formatına çevrilir
df["date_added"] = pd.to_datetime(df["date_added"], errors='coerce')
df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month

# 3. Film mi daha çok, dizi mi?
print("Film ve Dizi Sayısı:")
print(df["type"].value_counts())

# 4. Her yıl kaç içerik eklenmiş?
yillik_icerik = df["year_added"].value_counts().sort_index()
sns.lineplot(x=yillik_icerik.index, y=yillik_icerik.values)
plt.title("Yıllara Göre Eklenen İçerik Sayısı")
plt.xlabel("Yıl")
plt.ylabel("İçerik Sayısı")
plt.show()

# 5. Hangi yıl ne kadar içerik var (film & dizi ayrımı)
tur_yil = df.groupby(["year_added", "type"]).size().reset_index(name="count")
sns.lineplot(data=tur_yil, x="year_added", y="count", hue="type")
plt.title("Yıllara Göre Film ve Dizi Dağılımı")
plt.xlabel("Yıl")
plt.ylabel("İçerik Sayısı")
plt.legend(title="Tür")
plt.show()

# 6. Film sürelerini sayısala çevir
filmler = df[df["type"] == "Movie"].copy()
filmler = filmler.dropna(subset=["duration"])
filmler = filmler[filmler["duration"].str.contains("min")]
filmler["duration"] = filmler["duration"].str.replace(" min", "").astype(int)

# 7. Sürelerin yıllara göre dağılımı
sns.scatterplot(x="release_year", y="duration", data=filmler, alpha=0.3)
sns.regplot(x="release_year", y="duration", data=filmler, scatter=False, color="red")
plt.title("Yıllara Göre Film Süresi")
plt.xlabel("Yıl")
plt.ylabel("Süre (Dakika)")
plt.show()

# 8. Hangi ülke en çok içerik üretmiş
en_cok_ulkeler = df["country"].dropna().str.split(",").explode().str.strip().value_counts().head(10)
sns.barplot(x=en_cok_ulkeler.values, y=en_cok_ulkeler.index)
plt.title("En Çok İçerik Üreten Ülkeler")
plt.xlabel("İçerik Sayısı")
plt.ylabel("Ülke")
plt.show()

# 9. En çok içerik eklenen yıllar
en_yil = df["release_year"].value_counts().head(10)
sns.barplot(x=en_yil.index, y=en_yil.values)
plt.title("En Çok İçerik Çıkan Yıllar")
plt.xlabel("Yıl")
plt.ylabel("İçerik Sayısı")
plt.show()

# 10. En uzun 10 film
en_uzun = filmler.sort_values("duration", ascending=False).head(10)
print("En Uzun 10 Film:")
print(en_uzun[["title", "duration"]])

# 11. En yaygın türler
turler = df["listed_in"].str.split(",").explode().str.strip()
enyaygin = turler.value_counts().head(10)
sns.barplot(x=enyaygin.values, y=enyaygin.index)
plt.title("En Yaygın Türler")
plt.xlabel("İçerik Sayısı")
plt.ylabel("Tür")
plt.show()

# 12. Country sütununda yazım hataları var mı?
# Tüm ülkeleri listele
ulkeler = df["country"].dropna().str.split(",").explode().str.strip().value_counts()
print("Toplam ülke sayısı:", len(ulkeler))
print("Bazı ülke isimleri:")
print(ulkeler.head(20))

#outlier methods
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import chi2
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def z_score_method(df):
    mean = df["duration"].mean()
    std = df["duration"].std()
    df["z_score"] = (df["duration"] - mean) / std
    df["z_outlier"] = df["z_score"].abs() > 3
    return df

def iqr_method(df):
    Q1 = df["duration"].quantile(0.25)
    Q3 = df["duration"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df["iqr_outlier"] = (df["duration"] < lower) | (df["duration"] > upper)
    return df

def mahalanobis_method(df):
    data = df[["duration", "release_year"]]
    cov = EmpiricalCovariance().fit(data)
    md = cov.mahalanobis(data)
    threshold = chi2.ppf(0.99, df=2)
    df["mahal_outlier"] = md > threshold
    return df

def dbscan_method(df):
    X = df[["duration", "release_year"]]
    db = DBSCAN(eps=15, min_samples=5)
    labels = db.fit_predict(X)
    df["dbscan_label"] = labels
    df["dbscan_outlier"] = labels == -1
    return df

def isolation_forest_method(df):
    X = df[["duration", "release_year"]]
    model = IsolationForest(contamination=0.01, random_state=42)
    df["iso_outlier"] = model.fit_predict(X) == -1
    return df

def lof_method(df):
    X = df[["duration", "release_year"]]
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    labels = model.fit_predict(X)
    df["lof_outlier"] = labels == -1
    return df

#visualizations
import matplotlib.pyplot as plt
import seaborn as sns

def plot_duration_trend(df):
    sns.regplot(x="release_year", y="duration", data=df, scatter_kws={'alpha':0.4}, line_kws={"color": "red"})
    plt.title("Yıla Göre Film Süresi Trend Analizi")
    plt.xlabel("Yıl")
    plt.ylabel("Süre (Dakika)")
    plt.show()

def plot_outlier_distribution(df, method_col, title):
    sns.histplot(data=df, x="duration", hue=method_col, bins=30, kde=True)
    plt.title(title)
    plt.xlabel("Film Süresi")
    plt.ylabel("Frekans")
    plt.show()

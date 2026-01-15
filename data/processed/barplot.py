import matplotlib.pyplot as plt
import seaborn as sns

# Datos: reemplaza con tus datos reales
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
article_counts = [3, 7, 1, 10, 8, 25, 36, 47, 80, 119, 68]

# Estilo monocromático
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Barplot en escala de grises
sns.barplot(x=years, y=article_counts, color="gray")

# Etiquetas
plt.xlabel("Año de Publicación", fontsize=12)
plt.ylabel("Número de Artículos", fontsize=12)
plt.title('Distribución de Artículos sobre "AI" AND "EEG" en Scopus', fontsize=14)

# Añadir valores encima de las barras
for i, count in enumerate(article_counts):
    plt.text(i, count + 1, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig("barplot.png", dpi=300)
plt.close()
print("[INFO] Gráfico guardado como barplot.png")
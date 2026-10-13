import pandas as pd
import plotly.express as px
from scipy.cluster.hierarchy import linkage, leaves_list

df = pd.read_csv('./data/creadores.csv')
df_num = df.select_dtypes(include='number')

cov_matrix = df_num.cov()
fig_cov = px.imshow(cov_matrix, title="Matriz de Covarianza", text_auto=True, 
                    color_continuous_scale=['red', 'white', 'green'], color_continuous_midpoint=0)
fig_cov.write_html("covarianza.html")

corr_matrix = df_num.corr()
fig_corr = px.imshow(corr_matrix, title="Matriz de Correlación", text_auto=True, 
                     color_continuous_scale=['red', 'white', 'green'], color_continuous_midpoint=0)
fig_corr.write_html("correlacion.html")

order = leaves_list(linkage(corr_matrix, method='ward'))
corr_matrix_clustered = corr_matrix.iloc[order, order]
fig_corr_clust = px.imshow(corr_matrix_clustered, title="Matriz de Correlación Agrupada", text_auto=True, 
                           color_continuous_scale=['red', 'white', 'green'], color_continuous_midpoint=0)
fig_corr_clust.write_html("correlacion_agrupada.html")

import pandas as pd
import plotly.express as px

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

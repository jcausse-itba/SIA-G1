import pandas as pd
import numpy as np
import plotly.express as px

def standardize(df):
    return (df - df.mean()) / df.std()

def get_pca(df, q=2):
    df_std = standardize(df)
    cov_matrix = df_std.cov()
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    loadings = eigenvectors[:, :q]
    components = df_std.dot(loadings)
    components.columns = [f'PC{i+1}' for i in range(q)]
    
    loadings_df = pd.DataFrame(loadings, index=df.columns, columns=components.columns)
    return components, loadings_df

def create_boxplot(df_pca):
    fig = px.box(df_pca.drop(columns=['name'], errors='ignore'), title="Boxplot de Componentes Principales")
    fig.write_html("pca_boxplot.html")

def create_biplot(df_pca, loadings_df):
    fig = px.scatter(df_pca, x='PC1', y='PC2', text='name', title="Biplot PC1 vs PC2")
    fig.update_traces(textposition='top center')
    
    scale = max(df_pca['PC1'].max(), df_pca['PC2'].max()) / max(loadings_df['PC1'].max(), loadings_df['PC2'].max())
    
    for feature in loadings_df.index:
        x_end = loadings_df.loc[feature, 'PC1'] * scale
        y_end = loadings_df.loc[feature, 'PC2'] * scale
        
        fig.add_shape(type='line', x0=0, y0=0, x1=x_end, y1=y_end, line=dict(color='red', width=1.5))
        fig.add_annotation(x=x_end, y=y_end, text=feature, showarrow=False, font=dict(color='red'))
        
    fig.write_html("pca_biplot.html")

def create_index_plot(df_pca):
    df_pca = df_pca.sort_values(by='PC1', ascending=False).reset_index()
    fig = px.bar(df_pca, x='name', y='PC1', title="Índice de PC1")
    fig.write_html("pca_indice.html")

def main():
    df = pd.read_csv('./data/creadores.csv')
    df_num = df.select_dtypes(include='number')
    
    df_pca, loadings_df = get_pca(df_num, q=2)
    df_pca['name'] = df['name']
    
    create_boxplot(df_pca)
    create_biplot(df_pca, loadings_df)
    create_index_plot(df_pca)

if __name__ == "__main__":
    main()

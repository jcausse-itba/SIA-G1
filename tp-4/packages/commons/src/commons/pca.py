import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    fig = px.box(df_pca.drop(columns=['creator_id', 'name'], errors='ignore'), title="Boxplot de Componentes Principales")
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

def create_loadings_plot(loadings_df):
    
    for i, col in enumerate(loadings_df.columns):
        sorted_data = loadings_df[col].reindex(loadings_df[col].abs().sort_values(ascending=False).index)
        colors = ['green' if val > 0 else 'red' for val in sorted_data]
        
        fig.add_trace(go.Bar(
            x=sorted_data.index, 
            y=sorted_data, 
            name=col,
            visible=(i == 0),
            marker_color=colors
        ))
        
    buttons = []
    for i, col in enumerate(loadings_df.columns):
        visible = [False] * len(loadings_df.columns)
        visible[i] = True
        buttons.append(
            dict(
                label=col,
                method="update",
                args=[{"visible": visible}, 
                      {"title": f"Pesos de las características en {col}"}]
            )
        )
        
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.1, xanchor="left", y=1.1, yanchor="top")],
        title="Pesos de las características en PC1"
    )
    fig.write_html("pca_loadings.html")

def main():
    df = pd.read_csv('./data/creadores.csv')
    df_num = df.select_dtypes(include='number')
    
    df_pca, loadings_df = get_pca(df_num, q=df_num.shape[1])
    df_pca['creator_id'] = df['creator_id']
    df_pca['name'] = df['name']
    
    create_boxplot(df_pca)
    create_biplot(df_pca, loadings_df)
    create_index_plot(df_pca)
    create_loadings_plot(loadings_df)

    pca_cols = [c for c in df_pca.columns if c.startswith('PC')]
    df_pca[['creator_id', 'name'] + pca_cols].to_csv('./data/creadores_pca_all.csv', index=False)
    df_pca[['creator_id', 'name'] + pca_cols[:9]].to_csv('./data/creadores_pca_9.csv', index=False)

if __name__ == "__main__":
    main()

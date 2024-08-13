import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np
import seaborn as sns

# Função para carregar os dados
@st.cache_data
def load_data():
    heroes_info = pd.read_csv('heroes_information.csv',index_col='Unnamed: 0',na_values='-')
    powers = pd.read_csv('super_hero_powers.csv')
    most_common_powers = filter_powers(powers)
    powers = powers[['hero_names'] + list(most_common_powers)]
    return heroes_info, powers

#Função para filtrar os poderes mais comuns, com frequencia acima de 10%
def filter_powers(powers):
    power_freq = powers.mean()
    return power_freq[power_freq > 0.1].index


# Função para imputação de valores ausentes
def preprocess_data(df):
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

# Função para clustering
def perform_clustering(df, n_clusters=6):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    return clusters

# Função para calcular os centroids normalizados dos clusters
def calculate_cluster_centroids(df, clusters):
    df['Cluster'] = clusters
    cluster_centroids = df.groupby('Cluster').mean()
    scaler = MinMaxScaler()
    normalized_centroids = scaler.fit_transform(cluster_centroids)
    normalized_centroids = pd.DataFrame(normalized_centroids, columns=cluster_centroids.columns)
    return normalized_centroids

# Função para gerar o grafico radar dos clusters
def plot_radar_chart(normalized_centroids):
    labels = normalized_centroids.columns
    num_vars = len(labels)

    # Set up the radar plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, row in normalized_centroids.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'Centroid {i+1}')
        ax.fill(angles, values, alpha=0.25)

    # Draw one axe per variable + add labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    # Add a legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.show()

# Função para Naive Bayes
def predict_alignment(model, df, user_input):
    user_df = pd.DataFrame([user_input], columns=df.columns)
    return model.predict(user_df)[0]

# Função para regressão (previsão de peso)
def predict_weight(model, df, user_input):
    user_df = pd.DataFrame([user_input], columns=df.columns)
    return model.predict(user_df)[0]

# Função principal do app
def main():
    st.title("Superhero Data Explorer and ML Models")

    # Carregar os dados
    heroes_info, powers = load_data()

    st.sidebar.title("Menu")
    option = st.sidebar.selectbox("Escolha a seção", 
                                  ["Exploração de Dados", "Clustering", "Classificação do Alinhamento", "Previsão de Peso"])

    if option == "Exploração de Dados":
        st.header("Exploração de Dados")
        
        st.subheader("Visualização de Conjuntos de Dados")
        if st.checkbox("Mostrar heroes_information.csv"):
            st.write(heroes_info)
        
        if st.checkbox("Mostrar super_hero_powers.csv"):
            st.write(powers)
        
        st.subheader("Estatísticas Descritivas")
        st.write(heroes_info.describe())

        st.subheader("Distribuição de Variáveis")
        var = st.selectbox("Escolha a variável para visualizar", heroes_info.columns)
        st.bar_chart(heroes_info[var].value_counts())

        st.subheader("Filtragem de Super-Heróis")
        alignment = st.multiselect("Escolha o Alinhamento", heroes_info['Alignment'].unique())
        gender = st.multiselect("Escolha o Gênero", heroes_info['Gender'].unique())
        publisher = st.multiselect("Escolha a Editora", heroes_info['Publisher'].unique())
        
        filtered_data = heroes_info
        if alignment:
            filtered_data = filtered_data[filtered_data['Alignment'].isin(alignment)]
        if gender:
            filtered_data = filtered_data[filtered_data['Gender'].isin(gender)]
        if publisher:
            filtered_data = filtered_data[filtered_data['Publisher'].isin(publisher)]
        
        st.write(filtered_data)

    elif option == "Clustering":
        st.header("Clustering dos Super-Heróis")
        st.write("Aqui você pode visualizar os clusters de super-heróis com base em seus poderes.")
        
        # Preparar os dados para clustering
        powers_imputed = preprocess_data(powers.drop(columns=['hero_names']))
        clusters = perform_clustering(powers_imputed)
        
        st.subheader("Clusters Gerados")
        st.write(f"Clusters criados: {np.unique(clusters)}")
        powers.assign(Cluster=clusters)

        cluster_options = np.unique(clusters)
        selected_cluster = st.selectbox('Escolha o Cluster para Visualizar', cluster_options)
        st.write(powers[clusters == selected_cluster])

        st.subheader("Visualização dos Clusters")
        #plt.figure(figsize=(10, 6))
        #sns.scatterplot(x=powers_imputed[:, 0], y=powers_imputed[:, 1], hue=clusters, palette='viridis')
        #st.pyplot(plt)
        normalized_centroids = calculate_cluster_centroids(powers_imputed, clusters)
        
        labels = normalized_centroids.columns
        num_vars = len(labels)

        # Set up the radar plot
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Filter the centroids by cluster
        clusters_to_plot = [selected_cluster]  # Specify the clusters you want to plot

        for i, row in normalized_centroids.iterrows():
            if i in clusters_to_plot:
                values = row.values.flatten().tolist()
                values += values[:1]  # Complete the loop
                ax.plot(angles, values, linewidth=1, linestyle='solid')#, label=f'Centroid {i}')
                ax.fill(angles, values, alpha=0.25)

        # Draw one axe per variable + add labels
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)

        # Add a legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        #plt.show()
        st.pyplot(plt)

    elif option == "Classificação do Alinhamento":
        st.header("Classificação do Alinhamento (Bom ou Mau)")
        st.write("Use o modelo Naive Bayes para prever o alinhamento de um super-herói.")
        
        # Preparação dos dados
        heroes_filtered = heroes_info[heroes_info['Alignment'].isin(['good', 'bad'])]
        alignment_mapping = {'good': 1, 'bad': 0}
        heroes_filtered['Alignment'] = heroes_filtered['Alignment'].map(alignment_mapping)
        data = heroes_filtered.merge(powers, left_on='name', right_on='hero_names')
        data.dropna(subset='Weight', inplace=True)
        X = data.drop(columns=[ 'name', 'Alignment', 'hero_names','Publisher', 'Race', 'Gender', 'Eye color', 'Hair color', 'Skin color'])
        y = data['Alignment']
        #X_imputed = preprocess_data(X)

        # Treinar o modelo Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(X, y)
        
        st.subheader("Previsão de Alinhamento")
        user_input = {}
        for col in X.columns:
            if col == "Height":
                user_input[col] = st.slider(f"Escolha um valor para {col}", min_value=0, max_value=1000)
            elif col == "Weight":
                user_input[col] = st.slider(f"Escolha um valor para {col}", min_value=0, max_value=1000)
            else:
                user_input[col] = st.selectbox(f"Escolha um valor para {col}", options=list(X[col].unique()))
        
        prediction = predict_alignment(nb_model, X, user_input)
        st.write(f"O modelo prevê que o super-herói é: {'Bom' if prediction == 1 else 'Mau'}")

    elif option == "Previsão de Peso":
        st.header("Previsão de Peso de Super-Heróis")
        st.write("Use o modelo de regressão para prever o peso de um super-herói.")
        
        valid_weight_data = heroes_info[heroes_info['Weight'] > 0]
        data_reg = valid_weight_data.merge(powers, left_on='name', right_on='hero_names')
        X_reg = data_reg.drop(columns=['name', 'Weight', 'hero_names','Publisher', 'Race', 'Gender', 'Eye color', 'Hair color', 'Skin color', 'Alignment'])
        y_reg = data_reg['Weight']
        #X_reg_encoded = pd.get_dummies(X_reg, drop_first=True)
        #X_reg_imputed = preprocess_data(X_reg_encoded)

        # Treinar o modelo Random Forest Regressor
        rf_regressor = RandomForestRegressor(random_state=42, n_estimators=100)
        rf_regressor.fit(X_reg, y_reg)
        
        st.subheader("Previsão de Peso")
        user_input = {}
        for col in X_reg.columns:
            if col == "Height":
                user_input[col] = st.slider(f"Escolha um valor para {col}", min_value=0, max_value=1000)
            else:
                user_input[col] = st.selectbox(f"Escolha um valor para {col}", options=list(X_reg[col].unique()))
        
        prediction = predict_weight(rf_regressor, X_reg, user_input)
        st.write(f"O modelo prevê que o peso do super-herói é: {prediction:.2f} kg")

if __name__ == "__main__":
    main()

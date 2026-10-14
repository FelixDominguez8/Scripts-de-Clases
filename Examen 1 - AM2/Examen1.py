from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def main():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=50)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #4.

    pca = PCA(n_components=2, random_state=50)
    X_train_pca = pca.fit_transform(X_train)

    plt.figure()
    scatter = plt.scatter(X_train_pca[:, 0],
                        X_train_pca[:, 1],
                        c=y_train)
    plt.colorbar(scatter)
    plt.title("Visualizacion PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    tsne = TSNE(n_components=2,
                random_state=50,
                perplexity=30,
                learning_rate='auto',
                init='pca')

    X_train_tsne = tsne.fit_transform(X_train)

    plt.figure()
    scatter = plt.scatter(X_train_tsne[:, 0],
                        X_train_tsne[:, 1],
                        c=y_train)
    plt.colorbar(scatter)
    plt.title("Visualizacion t-SNE")
    plt.xlabel("Dimen 1")
    plt.ylabel("Dimen 2")
    plt.show()

    print(" ")
    print("¿Qué diferencias visuales se observan entre PCA y t-SNE? ¿Cuál representa mejor la estructura de los datos? ¿Cuál método es más adecuado si el objetivo fuera modelado predictivo y por qué?")
    print("R//= PCA forma mas in gran cluster en el medio sin ninguna forma aparente y segun como se ve, va en forma de gradiante de izquierda a derecha en donde en la izquierda se ven los valores mas oscuros (o bajos) y a la derecha los mas altos y coloridos")
    print("Con t-SNE se ve el mismo tipo de gradiente de izquierda a derecha, pero la forma de este cluster es mas definida como la de una c inversa")
    print("¿Cuál representa mejor la estructura de los datos?")
    print("R//= t-SNE debido a la forma mas definada del cluster que este brinda")
    print("¿Cuál método es más adecuado si el objetivo fuera modelado predictivo y por qué?")
    print("R//= PCA ya que este es un metodo lineal mientras que t-SNE no lo es")

    # 5.1

    regresion_elastic = ElasticNet(
        alpha=1,
        l1_ratio=0.5,
        random_state=50,
        max_iter=10000
    )

    regresion_elastic.fit(X_train, y_train)

    y_pred_elastic = regresion_elastic.predict(X_test)

    svr_lineal = SVR(
        kernel='linear',
        gamma=0.1,
        C=100,
        epsilon=0.1
    )

    svr_lineal.fit(X_train, y_train)

    y_pred_svr_lineal = svr_lineal.predict(X_test)

    svr_rbf = SVR(
        kernel='rbf',
        C=100,
        gamma=0.1,
        epsilon=0.1
    )

    svr_rbf.fit(X_train, y_train)

    y_pred_svr_rbf = svr_rbf.predict(X_test)

    #5.2

    rmse_elastic = np.sqrt(mean_squared_error(y_test, y_pred_elastic))
    r2_elastic = r2_score(y_test, y_pred_elastic)

    rmse_svr_lineal = np.sqrt(mean_squared_error(y_test, y_pred_svr_lineal))
    r2_svr_lineal = r2_score(y_test, y_pred_svr_lineal)

    rmse_svr_rbf = np.sqrt(mean_squared_error(y_test, y_pred_svr_rbf))
    r2_svr_rbf = r2_score(y_test, y_pred_svr_rbf)

    print("Regresion con Elastic Net - RMSE:", rmse_elastic, " R2:", r2_elastic)
    print("SVR Lineal - RMSE:", rmse_svr_lineal, " R2:", r2_svr_lineal)
    print("SVR con RBF - RMSE:", rmse_svr_rbf, " R2:", r2_svr_rbf)

    #5.3
    plt.figure()
    plt.scatter(y_test, y_pred_elastic)
    plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()])
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Predicho")
    plt.title("Regresion con Elastic NEt")
    plt.show()

    plt.figure()
    plt.scatter(y_test, y_pred_svr_lineal)
    plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()])
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Predicho")
    plt.title("SVR Lineal")
    plt.show()

    plt.figure()
    plt.scatter(y_test, y_pred_svr_rbf)
    plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()])
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Predicho")
    plt.title("SVR con RBF")
    plt.show()

    #5.4
    print(" ")
    print("Dados los siguientes resultados, ¿cuál modelo tiene mejor rendimiento para este dataset y por qué se debe esto?")
    print("R//= Segun las metricas SVR Lineal fue el mejor obteniendo ambos el menor RMSE y el mayor R2, esto posiblemente sea debido a que el dataset tiene una relacion lineal")
    print("¿qué nos indican estos resultados acerca de la relación entre las variables predictoras y el target en el dataset?")
    print("R//= Probablemente nos indique de que existe una relacion lienal entre las variables predictoras y el target del dataset")


if __name__ == "__main__":
    main()
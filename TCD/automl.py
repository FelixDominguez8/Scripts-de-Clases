from pycaret.datasets import get_data
from pycaret.classification import ClassificationExperiment

datos = get_data("iris")

exp = ClassificationExperiment()
exp.setup(data=datos, target="species", session_id=123)


mejor_modelo = exp.compare_models()

print("\n--- El mejor modelo seleccionado automáticamente ---")
print(mejor_modelo)
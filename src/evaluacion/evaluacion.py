#import ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

class Evaluator:
    def __init__(self, dataset):
        self.dataset = dataset

    def evaluate(self):
        """
        Evalúa los textos recuperados en función de su alineación con la consulta utilizando métricas de RAGAS.

        Returns:
            pandas.DataFrame: Resultados de la evaluación en un formato tabular.
        """
        result = evaluate(
            dataset=self.dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        return result.to_pandas()
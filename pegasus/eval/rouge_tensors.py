import tensorflow as tf
from rouge_score import rouge_scorer

_ROUGE_METRIC = "rouge1"  # change this depending on which metric you want


def evaluate(tensor1, tensor2):
    scorer = rouge_scorer.RougeScorer([_ROUGE_METRIC], use_stemmer=True)
    tensor1 = tensor1.numpy().decode("utf-8")  # decodes the target tensor
    tensor2 = tensor2.numpy().decode("utf-8")  # decodes the pred tensor

    r_score = scorer.score(tensor1, tensor2)  # calculates the rouge scores

    r_score_f1 = {el: 0 for el in list(r_score.keys())}  # set empty rouge dict
    r_score_f1.update({_ROUGE_METRIC: list(r_score[_ROUGE_METRIC])[2]})  # find F1 metric only

    return tf.Variable([[r_score_f1[_ROUGE_METRIC]]]).read_value()

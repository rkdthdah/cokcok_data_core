from .constant import CONST


def standardizeScore(score):
    """
    점수 정규화

    arameters:
    score (Float): SwingAnalysis의 analysis 함수로부터 산출된 score

    Returns:
    score (Float): 정규화 된 score
    """
    return score
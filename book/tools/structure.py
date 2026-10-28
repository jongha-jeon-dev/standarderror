"""The book's structure: nine parts, twenty-two chapters. Each chapter is
made of sections, and each section is one episode of the series (episode N
was written as book/sections/sNN.md)."""

PARTS = [
    ("I", "Reading a Score", [
        (1, "Scores, Error Bars and Comparisons", [1, 20]),
        (2, "What Cross-Validation Estimates", [2, 44, 36, 37]),
        (3, "Leakage", [3, 40, 45]),
        (4, "The Optimism of Search", [35, 67]),
        (5, "How Far More Data Goes", [4, 30, 68]),
    ]),
    ("II", "Models, Pushed", [
        (6, "Linear Models", [6, 7, 8, 9, 10]),
        (7, "Trees and Forests", [11, 13, 15]),
        (8, "Boosting, Averaging and Stacking", [14, 41, 42]),
        (9, "Training a Network", [21, 22, 23, 24, 25]),
        (10, "Generalisation in a Network", [26, 27, 28]),
    ]),
    ("III", "From Probabilities to Decisions", [
        (11, "Probabilities and Calibration", [5, 43, 34]),
        (12, "Thresholds and Costs", [46, 47, 69]),
        (13, "Causes and Feedback", [48, 49, 50]),
    ]),
    ("IV", "Reading What a Model Learned", [
        (14, "Importance and Explanation", [12, 16, 17, 18, 19]),
    ]),
    ("V", "Where the Rows Come From", [
        (15, "Missing, Selected and Shifted Data", [38, 39, 29]),
    ]),
    ("VI", "Without Labels", [
        (16, "Pictures of High-Dimensional Data", [31, 56]),
        (17, "Clusters", [32, 57, 58, 60]),
        (18, "Distance and Anomalies", [33, 59]),
    ]),
    ("VII", "Grading a Language Model", [
        (19, "Grading a Language Model", [51, 52, 53, 54, 55]),
    ]),
    ("VIII", "After the Model Ships", [
        (20, "Monitoring and Retraining", [61, 64]),
        (21, "Prediction Intervals and Fairness", [62, 63]),
    ]),
    ("IX", "The Book, Measured", [
        (22, "The Book, Measured", [65, 66, 70]),
    ]),
]

CHAPTERS = {c: (title, eps) for _, _, chs in PARTS for c, title, eps in chs}

# Episode N -> (chapter, section number within the chapter).
WHERE = {ep: (c, k + 1) for c, (_, eps) in CHAPTERS.items()
         for k, ep in enumerate(eps)}

assert sorted(WHERE) == list(range(1, 71)), sorted(set(range(1, 71)) - set(WHERE))
assert list(CHAPTERS) == list(range(1, 23))


def part_of(chapter: int) -> str:
    for num, _, chs in PARTS:
        if any(c == chapter for c, _, _ in chs):
            return num
    raise KeyError(chapter)

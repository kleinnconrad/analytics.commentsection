from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def corpus() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    articles = pd.DataFrame(
        {
            "ID_Article": [1, 2],
            "publishingDate": ["2015-06-01 08:00:00"] * 2,
            "Path": ["Newsroom/Politics/Migration", "Newsroom/Sports/Football"],
        }
    )
    rows = []
    for user, count in [(1, 10), (2, 10), (3, 5), (4, 1), (5, 10), (6, 5)]:
        for index in range(count):
            headline, body = "", f"Ordinary text {index}"
            if user == 1:
                headline, body = "Den Geflüchteten helfen", None
            elif user == 2:
                body = f"Wirtschaftsflüchtlinge: Beitrag {index}"
            elif user == 3:
                body = None
            elif user == 4:
                body = "Schutzsuchende"
            elif user == 5:
                body = f"{'Geflüchtete' if index < 5 else 'Asylflut'}: Beitrag {index}"
            votes = 100 if user == 4 else (10 if user == 6 else 0)
            rows.append(
                {
                    "ID_Post": len(rows) + 1,
                    "ID_User": user,
                    "ID_Article": 1 if user == 1 else index % 2 + 1,
                    "CreatedAt": f"2015-06-{index + 2:02d} 20:00:00",
                    "Headline": headline,
                    "Body": body,
                    "PositiveVotes": votes,
                    "NegativeVotes": votes // 2,
                }
            )
    return pd.DataFrame(rows), articles, pd.DataFrame({"ID_User": [1, 2, 99]})


@pytest.fixture
def data_dir(tmp_path: Path, corpus: tuple[pd.DataFrame, ...]) -> Path:
    for table, frame in zip(["Posts", "Articles", "Newspaper_Staff"], corpus, strict=True):
        frame.to_csv(tmp_path / f"{table}.csv", index=False)
    return tmp_path

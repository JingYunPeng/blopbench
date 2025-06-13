import sqlite3
import time

import pandas as pd
from typing import List, Dict, Optional, Union


class DB:
    def __init__(self, db_path: str = 'records.db'):
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_table()

    def _connect(self):
        self.conn = sqlite3.connect(self.db_path)

    def _create_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS record (
            function TEXT,
            algorithm TEXT,
            seed INTEGER,
            value REAL,
            PRIMARY KEY (function, algorithm, seed)
        )
        """
        self.conn.execute(create_table_sql)
        self.conn.commit()

    def insert_record(self, function: str, algorithm: str, seed: int, value: float) -> bool:
        try:
            self.conn.execute(
                "INSERT INTO record VALUES (?, ?, ?, ?)",
                (function, algorithm, seed, value)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def insert_many_records(self, records: List[Dict[str, Union[str, int, float]]]) -> int:
        count = 0
        for record in records:
            try:
                self.insert_record(
                    record['function'],
                    record['algorithm'],
                    record['seed'],
                    record['value']
                )
                count += 1
            except (KeyError, sqlite3.Error):
                continue
        return count

    def get_record(self, function: str, algorithm: str, seed: int) -> Optional[Dict[str, Union[str, int, float]]]:
        cursor = self.conn.execute(
            "SELECT * FROM record WHERE function=? AND algorithm=? AND seed=?",
            (function, algorithm, seed)
        )
        row = cursor.fetchone()
        if row:
            return {
                'function': row[0],
                'algorithm': row[1],
                'seed': row[2],
                'value': row[3]
            }
        return None

    def query_records(self, conditions: Dict[str, Union[str, int, float]] = None) -> List[
        Dict[str, Union[str, int, float]]]:
        if not conditions:
            sql = "SELECT * FROM record"
            params = ()
        else:
            where_clauses = []
            params = []
            for field, value in conditions.items():
                where_clauses.append(f"{field}=?")
                params.append(value)
            sql = f"SELECT * FROM record WHERE {' AND '.join(where_clauses)}"

        cursor = self.conn.execute(sql, params)
        return [
            {
                'function': row[0],
                'algorithm': row[1],
                'seed': row[2],
                'value': row[3]
            }
            for row in cursor.fetchall()
        ]

    def get_all_as_dataframe(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM record", self.conn)

    def update_record_value(self, function: str, algorithm: str, seed: int, new_value: float) -> bool:
        cursor = self.conn.execute(
            "UPDATE record SET value=? WHERE function=? AND algorithm=? AND seed=?",
            (new_value, function, algorithm, seed)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_record(self, function: str, algorithm: str, seed: int) -> bool:
        cursor = self.conn.execute(
            "DELETE FROM record WHERE function=? AND algorithm=? AND seed=?",
            (function, algorithm, seed)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def db(self):
        return self

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


record = DB()


def save(function: str, algorithm: str, seed: int, value: float):
    with record.db() as db:
        db.insert_record(function, algorithm, seed, value)


def load_all():
    with record.db() as db:
        return db.get_all_as_dataframe()


if __name__ == '__main__':
    with record.db() as db:
        df = db.get_all_as_dataframe()
        result = df.groupby('function')['value'].agg([
            ('median', 'median'),  # 中位数
            ('IQR', lambda x: x.quantile(0.75) - x.quantile(0.25))  # 四分位距
        ]).reset_index()
        print(result)

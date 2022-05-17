import uuid
import hashlib
import sqlite3


class DBManager:
    """
    DBManager: A manager for the database (SQLite) managements.
    """

    def __init__(self, db_file):
        """
        Create a database connection to a SQLite database.
        :param db_file: the database file location.
        """
        try:
            self.conn = sqlite3.connect(db_file)
            self.cursor = self.conn.cursor()
            self.cursor.execute("CREATE TABLE AL_POOL (MD5 TEXT, UUID TEXT, DATA TEXT);")
        except Exception as e:
            print(e)

    def close_connection(self):
        self.conn.close()

    def connect(self, db_file):
        self.conn = sqlite3.connect(db_file)

    def insert_record(self, data_pth, skip=True):
        random_uuid = str(uuid.uuid4())
        file_id = hashlib.md5(data_pth.encode('utf-8')).hexdigest()
        if skip:
            if not self.check_row(file_id):
                self.cursor.execute("INSERT INTO AL_POOL VALUES (?, ?, ?);", (file_id, random_uuid, data_pth))
        else:
            self.cursor.execute("INSERT INTO AL_POOL VALUES (?, ?, ?);", (file_id, random_uuid, data_pth))

        self.conn.commit()

    def read_records(self):
        return self.cursor.execute("SELECT MD5, UUID, DATA FROM AL_POOL").fetchall()

    def check_row(self, md5):
        result = self.cursor.execute("SELECT EXISTS(SELECT 1 FROM AL_POOL WHERE MD5 = ?);", (md5,)).fetchall()
        return result[0][0] is not 0


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

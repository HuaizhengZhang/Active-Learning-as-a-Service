"""
Toolkit for data operations.
@author huangyz0918 (huangyz0918@gmail.com)
@date 28/05/2022
"""

import uuid
import hashlib
import sqlite3
import boto3
import urllib.request
import numpy as np
from abc import ABC, abstractmethod


class Downloader(ABC):
    """
    Downloader: the base data downloader class.
    """

    @abstractmethod
    def download(self, filename, **kwargs):
        pass


class UrlDownloader(Downloader):
    """
    Url Downloader: the data downloader of given urls.
    """

    def download(self, filename, url=None):
        if url:
            urllib.request.urlretrieve(url, filename)
            return filename
        else:
            return None


class S3Downloader(Downloader):
    """
    S3 Downloader: the data downloader of AWS S3.
    """

    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name):
        session = boto3.session.Session()
        self.bucket = session.resource('s3',
                                       aws_access_key_id=aws_access_key_id,
                                       aws_secret_access_key=aws_secret_access_key).Bucket(bucket_name)

    def download(self, filename, object_name=None):
        self.bucket.download_file(object_name, filename)
        return filename


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
            sqlite3.register_adapter(np.ndarray, adapt_array)
            sqlite3.register_converter("ARRAY", convert_array)
            self.conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute("CREATE TABLE IF NOT EXISTS AL_POOL (MD5 TEXT, UUID TEXT, DATA TEXT, INFER ARRAY);")
        except Exception as e:
            print(e)

    def close_connection(self):
        """
        Close the connection of SQLite.
        """
        self.conn.close()

    def connect(self, db_file):
        """
        Establish the connection of SQLite.
        :param db_file: the database file location.
        """
        self.conn = sqlite3.connect(db_file)

    def insert_record(self, data_pth, infer_result=None, skip=True):
        """
        Insert the record to SQLite.
        :param data_pth: the path of inference request file.
        :param skip: skip the file with the same MD5.
        :param infer_result: the inference result according to the input data.
        """
        random_uuid = str(uuid.uuid4())
        file_id = hashlib.md5(data_pth.encode('utf-8')).hexdigest()
        if skip:
            if not self.check_row(file_id):
                self.cursor.execute("INSERT INTO AL_POOL VALUES (?, ?, ?, ?);",
                                    (file_id, random_uuid, data_pth, infer_result))
        else:
            self.cursor.execute("INSERT INTO AL_POOL VALUES (?, ?, ?, ?);",
                                (file_id, random_uuid, data_pth, infer_result))

        self.conn.commit()

    def update_inference(self, data_uuid, infer_result):
        """
        Update the inference result according to the given UUID.
        """
        self.cursor.execute("UPDATE AL_POOL SET INFER = ? WHERE UUID = ?", (infer_result, data_uuid))
        self.conn.commit()

    def update_inference_md5(self, data_md5, infer_result):
        """
        Update the inference result according to the given MD5.
        """
        self.cursor.execute("UPDATE AL_POOL SET INFER = ? WHERE MD5 = ?", (infer_result, data_md5))
        self.conn.commit()

    def read_records(self, with_infer=False):
        """
        Read all records from the current database.
        """
        if with_infer:
            return self.cursor.execute("SELECT MD5, UUID, DATA, INFER FROM AL_POOL").fetchall()
        else:
            return self.cursor.execute("SELECT MD5, UUID, DATA FROM AL_POOL").fetchall()

    def get_rows(self, inferred=True):
        """
        Get all the rows with inference results.
        """
        if inferred:
            return self.cursor.execute("SELECT MD5, UUID, DATA, INFER FROM AL_POOL WHERE INFER IS NOT NULL").fetchall()
        else:
            return self.cursor.execute("SELECT MD5, UUID, DATA, INFER FROM AL_POOL WHERE INFER IS NULL").fetchall()

    def check_row(self, md5):
        """
        Check is a row exists in the dataset by MD5.
        """
        result = self.cursor.execute("SELECT EXISTS(SELECT 1 FROM AL_POOL WHERE MD5 = ?);", (md5,)).fetchall()
        return result[0][0] != 0


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def adapt_array(arr):
    """
    Adapt the numpy.array to SQLite.
    """
    return arr.tobytes()


def convert_array(text):
    """
    Convert the SQLite bytes to numpy.array.
    """
    return np.frombuffer(text)

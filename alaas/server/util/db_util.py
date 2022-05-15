"""
Database Manager for Experiment Records.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 17/08/2020
"""

import time

import docker
import pymongo


class DBManager:
    """
    DBManager: A manager for the database (mongoDB) managements.
    """

    def __init__(self, db_name, container_name, ports):
        """
        Parameters.
        :param db_name: the database name.
        :param container_name: the docker container of mongo's name.
        :param ports: the ports mapping.
        """
        self.db_name = db_name
        self.ports = ports
        self.container_name = container_name
        self.docker_client = docker.from_env()
        self.db_container = None

    def start_db_container(self, timeout):
        """
        Start the Docker container.
        :param timeout: the time to wait the docker started.
        :return: None
        """
        self.db_container = self.docker_client.containers.run(
            image='mongo', remove=True, detach=True, ports=self.ports,
            name=self.container_name
        )
        time.sleep(timeout)

    def stop_db_container(self):
        """
        Stop the database container.
        :return: None
        """
        self.db_container.stop()

    def get_container_status(self):
        """
        check the docker container's status, could be 'running' or 'exited'.
        :return: None
        """
        return self.db_container.status

    @staticmethod
    def get_mongo_client(address='mongodb://localhost:27017/'):
        """
        Get the mongo Python client
        :param address: the database service address, contains ports and host.
        :return: client.
        """
        return pymongo.MongoClient(address)

    @staticmethod
    def create_db(name='bmk', address='mongodb://localhost:27017/', username='hyz', password='123456'):
        """
        Create the database to store the benchmarking results.
        :param name: the database's name.
        :param address: the database service address, contains ports and host.
        :param username: the database user name.
        :param password: the database password.
        :return: a new table.
        """
        mongo_client = pymongo.MongoClient(address,
                                           username=username,
                                           password=password,
                                           authSource=name,
                                           authMechanism='SCRAM-SHA-256')
        return mongo_client[name]

    @staticmethod
    def create_db_without_auth(name='pycontinual', address='mongodb://localhost:27017/'):
        """
        Create the database to store the benchmarking results.
        :param name: the database's name.
        :param address: the database service address, contains ports and host.
        :return: a new table.
        """
        mongo_client = pymongo.MongoClient(address,
                                           authSource=name,
                                           authMechanism='SCRAM-SHA-256')
        return mongo_client[name]

    @staticmethod
    def create_doc(database, doc_name='experiment'):
        return database[doc_name]

    @staticmethod
    def insert_record(db_doc, record: dict):
        """
        Insert single record into the database.
        :param db_doc: the document of the created database.
        :param record: the single record needs to be inserted.
        :return:
        """
        return db_doc.insert_one(record).inserted_id

    @staticmethod
    def insert_record_list(db_doc, record_list: list):
        """
        Insert records into the database.
        :param db_doc: the document of the created database.
        :param record_list: the records need to be inserted.
        :return:
        """
        return db_doc.insert_many(record_list).inserted_ids

    @staticmethod
    def delete_one_record(db_doc, query):
        """
        Query the records from the database table, and delete it.
        :param db_doc: the document of database.
        :param query: the query object (e.g., {"address": "Park Lane 38"}).
        :return: the deleted record id.
        """
        return db_doc.delete_one(query)

    @staticmethod
    def delete_records(db_doc, query):
        """
        Query the records from the database table, and delete them.
        :param db_doc: the document of database.
        :param query: the query object (e.g., {"address": "Park Lane 38"}).
        :return: the deleted records id.
        """
        return db_doc.delete_many(query)

    @staticmethod
    def delete_db_set(db_doc):
        """
        Delete the specific database table.
        :param db_doc: the documents of database you want to delete.
        """
        return db_doc.drop()

    @staticmethod
    def delete_db(client: pymongo.MongoClient, db_name: str):
        """
        Delete the specific database table.
        :param db_name: the database you want to delete.
        :param client: the mongo client.
        """
        client.drop_database(db_name)

    @staticmethod
    def update_one_record(db_doc, query, new_values):
        """
        Update the single record.
        :param db_doc: the document of database.
        :param query: the query object (e.g., {"address": "Park Lane 38"}).
        :param new_values: the new value objects (e.g., {"$set":{"address": "Canyon 123"}}).
        :return: Updated record's id.
        """
        return db_doc.update_one(query, new_values)

    @staticmethod
    def update_records(db_doc, query, new_values):
        """
        Update the single record.
        :param db_doc: the document of database.
        :param query: the query object (e.g., {"address": "Park Lane 38"}).
        :param new_values: the new value objects (e.g., {"$set":{"address": "Canyon 123"}}).
        :return: Updated record's id.
        """
        return db_doc.update_many(query, new_values)

    @staticmethod
    def query_record(db_doc, query):
        """
        Query the records from the database table.
        :param db_doc: the document of database.
        :param query: the query object (e.g., {"address": "Park Lane 38"}).
        :return: the matched records.
        """
        return db_doc.find(query)

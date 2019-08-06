import os


class Path(object):
    @staticmethod
    def db_root_dir(database=''):
        db_root = '/path/to/db/dir/'
        if database == 'PASCAL_MT':
            return os.path.join(db_root, 'PASCAL_MT')
        elif database == 'NYUD_MT':
            return os.path.join(db_root, 'NYUD_MT')
        elif not database:
            return db_root
        else:
            raise NotImplementedError

    @staticmethod
    def save_root_dir():
        return './'

    @staticmethod
    def exp_dir():
        return './'

    @staticmethod
    def models_dir():
        return '/path/to/pre-trained/models/'

    @staticmethod
    def cluster_log_dir():
        return '/checkpoint/your_username/'

    @staticmethod
    def seism_root_dir():
        # For edge detection evaluation
        return '/path/to/seism'

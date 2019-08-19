# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os


class Path(object):
    """
    User-specific path configuration.
    Please complete the /path/to/* paths to point into valid directories.
    """
    @staticmethod
    def db_root_dir(database=''):
        db_root = '/path/to/databases'

        db_names = {'PASCAL_MT', 'NYUD_MT', 'BSDS500', 'NYUD_raw',
                    'PASCAL', 'COCO', 'FSV', 'MSRA10K', 'PASCAL-S'}

        if database in db_names:
            return os.path.join(db_root, database)
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
    def seism_root_dir():
        # For edge detection evaluation (optional)
        return '/path/to/seism'

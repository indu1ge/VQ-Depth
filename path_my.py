class Path(object):
    @staticmethod
    def get_path_of(name):
        if name == "kitti":
            return 'data/kitti'
        elif name == 'cityscapes':
            return 'data/cityscapes'
        else:
            raise NotImplementedError
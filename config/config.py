import yaml


class Config:
    def __init__(self):
        self.settings = {}
        self.files = {
            'general': 'config/general.yaml',
            'problems': 'config/problems.yaml',
            'optimizers': 'config/optimizers.yaml',
            'benchmarks': 'config/benchmarks.yaml'
        }

        self.get_config()

    def get_config(self):
        for k, v, in self.files.items():
            with open(v, 'r') as stream:
                try:
                    self.settings[k] = yaml.safe_load(stream)
                except yaml.YAMLError as e:
                    print(e)

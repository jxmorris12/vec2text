import json

import transformers

NEW_ATTRIBUTES = {
    "embedder_torch_dtype": "float32",
}


class InversionConfig(transformers.configuration_utils.PretrainedConfig):
    """We create a dummy configuration class that will just set properties
    based on whatever kwargs we pass in.

    When this class is initialized (see experiments.py) we pass in the
    union of all data, model, and training args, all of which should
    get saved to the config json.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                json.dumps(value)
                setattr(self, key, value)
            except TypeError:
                # value was not JSON-serializable, skip
                continue
        super().__init__()
        self.custom_embedder_name = "gtr-base"

    def __getattribute__(self, key):
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            if key in NEW_ATTRIBUTES:
                return NEW_ATTRIBUTES[key]
            else:
                raise e

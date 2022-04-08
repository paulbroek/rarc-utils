""" 
    This is what api_keys.py should look like
""" 
import json
from abc import ABCMeta

class blueprint_keys(metaclass=ABCMeta):
    """ config files can be implemented by setting class attributes:
        login = 'user'
        pass =  'somepass'
    """

    @classmethod
    def property_dict(cls):
        #print(f'{dir(cls)=}')

        return dict((k,v) for k,v in cls.__dict__.items() if not(k.startswith('_') or k.startswith('__') or callable(v))) 

    @classmethod
    def json(cls):
        """ print json config, to be saved in this folder """
        #print(cls.__name__)
        return json.dumps({cls.__name__: cls.property_dict()}, indent=4, sort_keys=True)

    @classmethod
    def save_json(cls):
        with open(f"{cls.__name__}.json", "w") as outfile:
            json.dump(json.loads(cls.json()), outfile, ensure_ascii=False, indent=4)

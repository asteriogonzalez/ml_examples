from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from aiml import *


def test_checkpoint():
    "Test basic checkpoint features"
    chp = Checkpoint()
    chp['foo'] = 'bar'
    # chp.save()
    filename = chp.filename
    del chp

    chp2 = Checkpoint(filename)
    assert chp2['foo'] == 'bar'

def test_template_for_studies():
    "A simple template proposal for create a *study* test case"
    result = Checkpoint()

    def func():
        result['foo'] = 'buzz'
        return result

    result = result or func()

    print result



if __name__ == '__main__':
    test_checkpoint()
    test_template_for_studies()


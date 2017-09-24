import pytest
import time



@pytest.mark.study
# @pytest.mark.dependency(depends=['bar'])
# @pytest.mark.incremental
def test_study_1():
    time.sleep(0.2)
    results = 'Hello World!!!'
    print "Study 1 results: %s" % (results)

@pytest.mark.study
# @pytest.mark.incremental
def test_study_2():
    time.sleep(0.3)
    results = 'Hello World again!!!'
    print "Study 2 results: %s" % (results)


# @pytest.mark.dependency(name='foo')
# @pytest.mark.incremental
@pytest.mark.study
def test_foo():
    time.sleep(0.1)
    print "inside foo test!"
    assert True

# @pytest.mark.dependency(name="bar", depends=['foo'])
# @pytest.mark.incremental
@pytest.mark.study
def test_bar():
    time.sleep(0.15)
    print "inside bar test!"
    assert True


def test_independent():
    "An isolated test"
    time.sleep(0.05)

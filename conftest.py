# import wingdbstub
import pytest
import re

reg_study = re.compile(r'.*_study_.*', re.DOTALL|re.I)


def pytest_report_header(config):
    return "Hello World!!"


def pytest_runtest_teardown(item):
    "Called for teardown when item has been tested"

    foo = 1



def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    # option_value = metafunc.config.option.name
    # if 'name' in metafunc.fixturenames and option_value is not None:
        # metafunc.parametrize("name", [option_value])
    pass

def pytest_cmdline_preparse(args):
    # if 'xdist' in sys.modules: # pytest-xdist plugin
        # import multiprocessing
        # num = max(multiprocessing.cpu_count() / 2, 1)
        # args[:] = ["-n", str(num)] + args
    pass

#------------------------------------------
# Custom python test collector
#------------------------------------------
def pytest_collect_file(path, parent):
    # print ("me:\tchecking %s" % path.basename)

    # ext = path.ext
    # if ext == ".py":
        # if not parent.session.isinitpath(path):
            # for pat in parent.config.getini('python_files'):
                # if path.fnmatch(pat):
                    # break
            # else:
                # return
        # ihook = parent.session.gethookproxy(path)
        # return ihook.pytest_pycollect_makemodule(path=path, parent=parent)
    pass


#------------------------------------------
# Skip studio tests
#------------------------------------------
def pytest_addoption(parser):
    parser.addoption("--runstudy", action="store_true",
                     default=False, help="run studio processes")


def pytest_collection_modifyitems(config, items):
    """Remove all study tests if --runstudy is not selected
    and reorder the study dependences to be executed incrementaly
    so any failed study test will abort the complete sequence.

    - Mark a test with @pytest.mark.study to consider part of a study.
    - Mark a test with @pytest.mark.study and named 'test_study_xxxx()'
      to be executed at the end when all previous test study functions
      are passed.
    """
    # check if studio tests myst be skipped
    if config.getoption("--runstudy"):
        # --runstudy given in cli: do not skip study tests and
        # set @pytest.hookimpl(trylast=True) by default for all studio tests
        regular_tests = []
        study_dependences = []
        study_tests = []
        incremental = pytest.mark.incremental()
        for item in items:
            if "study" in item.keywords:
                # mark all study tests as incremental
                print "%s.incremental" % item.name
                item.add_marker(incremental)

                # move the long real study test to the end in executio queue
                if reg_study.match(item.name):
                    print "%s.trylast = True" % item.name
                    item.keywords['pytest_impl'] =  {'hookwrapper': False,
                                 'optionalhook': False,
                                 'tryfirst': False,
                                 'trylast': True}
                    study_tests.append(item)
                else:
                    study_dependences.append(item)
            else:
                regular_tests.append(item)

        # reorder tests IN-PLACE
        items[:] = regular_tests + study_dependences + study_tests
        return

    skip_test = pytest.mark.skip(reason="need --runstudy option to run")
    for item in items:
        if "study" in item.keywords:
            item.add_marker(skip_test)

def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item

def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" %previousfailed.name)

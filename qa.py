#!/usr/bin/env python
"""Help tool for code analisys tasks
"""
import os
import subprocess
import re

ROOT = os.path.abspath('.')
PROJECT_NAME = os.path.basename(ROOT)

CA = os.path.join(ROOT, 'ca')
# pylint has a problem with dependence paths output
# because duplicate the last basename path
# so we assume dot files will be generating in ca/ca/*.dot
DEP = os.path.join(CA, 'dependences')
CA_WORKARROUND = os.path.join(CA, 'ca')
UML = os.path.join(CA, 'uml')


def makedirs(*folders):
    "Create folders"
    for name in folders:
        # make the output directory for dependences graphs
        if not os.path.exists(name):
            os.makedirs(name)


def find_files(top, pattern):
    "Return all recursive files that match a regexp pattern"
    result = []
    regexp = re.compile(pattern)
    top += os.path.sep
    for root, _, files in os.walk(top):
        for name in files:
            if regexp.match(name):
                name = os.path.join(root, name)
                name = name.split(top)[-1]
                result.append(name)
    return result


def move_files(pattern, where, top='.'):
    "Move matching files to another location"
    for old in find_files(top, pattern):
        new = os.path.join(where, os.path.basename(old))
        os.renames(old, new)


def run_pylint(files):
    """Make a global pylint execution with all files.
    Makes also module dependence diagrams and spell-cheking
    """
    # pylint return codes
    # 1: file missing
    # 20, 30: seems ok
    cmd = ['pylint', ]
    cmd.extend(files)
    print cmd
    pylint = subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    stdout, stderr = pylint.communicate()

    with file(os.path.join(CA, 'pylint.out.txt'), 'w') as report:
        report.write(stdout)

    dot_files = find_files(ROOT, r'.*dep_\S+\.dot$')
    for name in dot_files:
        cmd = ['dot', '-Tpng', '-O', name]
        print cmd
        dot = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
        stdout, stderr = dot.communicate()

    move_files(r'dep\S*\.(dot|png)$', DEP)


def run_pyreverse(files):
    "Create UML diagrams for each python file"
    for name in files:
        # pyreverse -mn -a1 -s1 -f ALL -o png ex2.py -p
        cmd = ['pyreverse', '-mn', '-a1', '-s1',
               '-fALL', '-opng']

        basename = os.path.basename(name)
        cmd.append('-p%s' % basename)
        cmd.append(name)
        print cmd
        pyreverse = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        stdout, stderr = pyreverse.communicate()

    move_files(r'classes_\S*\.png$', UML)


def main():
    """Performs code analysis tasks like pylint, uml diagram,
    module dependences, spell-checking, etc
    """
    makedirs(CA, CA_WORKARROUND, UML)

    PYTHON_FILES = find_files(ROOT, r'.*\.py$')
    run_pylint(PYTHON_FILES)
    run_pyreverse(PYTHON_FILES)


if __name__ == '__main__':
    main()

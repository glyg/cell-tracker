.PHONY: help init build clean test coverage doc push_doc

help:
	@echo "Please use make <target> where <target> is one of"
	@echo "    init           : init and pull all submodules"
	@echo "    build          : build extensions (not needed yet)"
	@echo "    clean          : clean current repository"
	@echo "    test           : run tests"
	@echo "    coverage       : run tests and check code coverage"
	@echo "    push_doc       : push dev documentation to http://bnoi.github.io/scikit-tracker/dev/"

init:
	git submodule update --init

build:
	python setup.py build_ext --inplace

clean:
	find . -name "*.so" -exec rm -rf {} \;
	find . -name "*.pyc" -exec rm -rf {} \;
	find . -depth -name "__pycache__" -type d -exec rm -rf '{}' \;
	rm -rf build/ dist/ scikit_tracker.egg-info/

test:
	nosetests cell_tracker -v

coverage:
	nosetests cell_tracker --with-coverage --cover-package=sktracker -v

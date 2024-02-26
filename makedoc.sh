#!/bin/bash

pdoc --html ../Cynet/ -o docs/ -c latex_math=True -f --template-dir docs/dark_templates

cp -r docs/cynet/* docs

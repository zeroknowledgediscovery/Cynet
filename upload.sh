#!/bin/bash

COMMENT=" updated "

ONLYTEST=0


if [ $# -gt 0 ] ; then
    COMMENT=$1
else
    echo "USAGE: ./upload.sh <comment> <only_test>"
    echo "if only_test is 1, then upload to only test"
    exit
fi
if [ $# -gt 1 ] ; then
    ONLYTEST=$2
fi

a=$((`awk -F= '{print $2}' version.py  | sed "s/\s*'//g" | awk -F. '{print $NF}'` + 1))
VERSION=`awk -F= '{print $2}' version.py  | sed "s/\s*'//g" | awk -F. 'BEGIN{OFS="."}{print $1,$2}'`.$a


echo __version__ = \'$VERSION\'
#exit

echo __version__ = \'$VERSION\' > version.py

../gitscripts_/gitscript.sh $COMMENT
git tag $VERSION -m $COMMENT
git push --tags
python setup.py sdist upload -r test

if [ $ONLYTEST -eq 0 ] ; then
    python setup.py sdist upload -r pypi
fi

#!/bin/bash

set -euo pipefail

git clone -b gh-pages "https://loda-bot:${LODA_BOT_TOKEN}@github.com/loda-lang/loda-python.git" $HOME/gh-pages
git -C $HOME/gh-pages config user.name "LODA Bot"
git -C $HOME/gh-pages config user.email "${LODA_BOT_EMAIL}"
rm -r $HOME/gh-pages/docs
mv html/loda $HOME/gh-pages/docs
rmdir html
git -C $HOME/gh-pages add docs
git -C $HOME/gh-pages commit -m "update docs"
git -C $HOME/gh-pages push
rm -rf $HOME/gh-pages

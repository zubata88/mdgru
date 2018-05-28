#!/usr/bin/env bash

GIT_DIR=$(git rev-parse --git-dir)

echo "installing hooks..."

ln -s ../../scripts/pre-commit.sh $GIT_DIR/hooks/pre-commit
ln -s ../../scripts/pre-push.sh $GIT_DIR/hooks/pre-push

echo "Done!"

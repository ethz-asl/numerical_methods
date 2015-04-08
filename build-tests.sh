#!/bin/bash

CURRENT_PATH="`dirname \"$0\"`"
if [ ! -d "$CURRENT_PATH/bin" ]; then
  mkdir $CURRENT_PATH/bin
fi

g++ -I $CURRENT_PATH/include/ $CURRENT_PATH/test/test-integration-methods.cc /usr/lib/libgtest.a /usr/lib/libgtest_main.a -lglog -lpthread -o $CURRENT_PATH/bin/test-integration-methods -std=c++11
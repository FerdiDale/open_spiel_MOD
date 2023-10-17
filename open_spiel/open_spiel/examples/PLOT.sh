#!/bin/bash

filelist=$(ls | cat | grep 'gnu .*')
filelist=$(echo "$filelist" | sed -e "s/^/\"/" -e "s/$/\"/")
echo "$filelist" | awk '{$0=$0; system("gnuplot <" $0)}'

#!/bin/bash

grep TRIPLET | grep -v _im= | grep -v _disc= | perl -n -e's/\[MAIN\]\[0\]conf #\d+ mass=(-*[\.\d]+)( DEFAULT_SEMWALL)? TRIPLET/$1/g; chomp; s/=//g; s/[A-Z_]* TRIPLET//g; print "$_\n" unless ($_=~/^\s?$/);' | perl -n -e'$a=int(($.-1)/17); print "$a $_";'

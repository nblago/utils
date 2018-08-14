#/bin/bash

echo $0

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters."
    echo "Usage: python_2to3_converter.sh old_python2.X.py new_python_3.X.py"

else
    if [ $1 = $2 ]; then
	echo "File names shall be different"
	exit
    fi
    #Run the substitution for the print
    echo "from __future__ import print_function" > $2
    cat  $1 | sed 's/print \(.*\)/print \(\1\) /' >> $2
fi

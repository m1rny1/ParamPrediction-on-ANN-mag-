# ParamPrediction-on-ANN-mag-
Parameter prediction as time-series forecasting method based on convolutional ANN. 
Program has no GUI so you can use it in CMD.
As input file you can use XLSX with more than 2 parameters in columns and more than 20 timesteps in rows.

Commandline arguments:
 -h, --help            show this help message and exit
 -f FILE, --file FILE  Path to xlsx file with data.
 -s SHEET [SHEET ...], --sheet SHEET [SHEET ...]
                       Sheet number. By default using first sheet in file
 -e EPOCH [EPOCH ...], --epoch EPOCH [EPOCH ...]
                       Max amount of training. By default using 100 epochs
 -w WINDOW [WINDOW ...], --window WINDOW [WINDOW ...]
                       Data window approved values size. By default using 9
 -c CASTSIZE [CASTSIZE ...], --castsize CASTSIZE [CASTSIZE ...]
                       Number of forecasting steps. By default using 2
 -v VERBOSE [VERBOSE ...], --verbose VERBOSE [VERBOSE ...]
                       Level of describe detailing. By default using 1
 -i IGNORETRAININGERRORS, --ignoretrainingerrors IGNORETRAININGERRORS
                       Ignore training errors and complete full train amount. By default using False

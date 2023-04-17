# ParamPrediction-on-ANN-mag-<br>
Parameter prediction as time-series forecasting method based on convolutional ANN. <br>
Program has no GUI so you can use it in CMD.<br>
As input file you can use XLSX with more than 2 parameters in columns and more than 20 timesteps in rows.<br><br>


Commandline arguments:
```
	-h, --help            show this help message and exit
	-f FILE, --file FILE  Path to xlsx file with data.<br>
	-s SHEET [SHEET ...], --sheet SHEET [SHEET ...]
                       Sheet number. By default using first sheet in file<br>
	-e EPOCH [EPOCH ...], --epoch EPOCH [EPOCH ...]
                       Max amount of training. By default using 100 epochs<br>
	-w WINDOW [WINDOW ...], --window WINDOW [WINDOW ...]
                       Data window approved values size. By default using 9<br>
	-c CASTSIZE [CASTSIZE ...], --castsize CASTSIZE [CASTSIZE ...]
                       Number of forecasting steps. By default using 2<br>
	-v VERBOSE [VERBOSE ...], --verbose VERBOSE [VERBOSE ...]
                       Level of describe detailing. By default using 1<br>
	-i IGNORETRAININGERRORS, --ignoretrainingerrors IGNORETRAININGERRORS
                       Ignore training errors and complete full train amount. By default using False
```

executable            = Condor_run.sh
output                = logs/Condor_run.$(ClusterId).$(ProcId).out
error                 = logs/Condor_run.$(ClusterId).$(ProcId).err
log                   = logs/Condor_run.$(ClusterId).log
+JobBatchName         = "Condor_run"


# # ------ MagDown --------
## Change the numbers to the range of files you want to process, 
## e.g.  0 309 for all MagDown
## or    0 332 for all MagUp
arguments=  $(Item) MagDown
queue from seq 3 309 |

# # # ------ MagUp --------
arguments=  $(Item) MagUp
queue from seq 3 332 |

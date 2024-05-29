# Surver Room Surveilence System: Phase 1 - Audio Monitoring

DESCRIPTION:
These are two scripts that will be uploaded onto the office and lab NUCs - after some adjustments.

What does lab.py do?
- Captures audio from laptop mic
- Processed the data
- Sends processed data to a virtual queue
- Displays processed data with some options for the data format

What does office.py do?
- Reads the data from the virtual queue (if on the same network as lab.py)
- Displays processed data with some options for the data format

INSTRUCTIONS:
Run both scripts by typing the following commands on seperate command line windows:
<python .\lab.py -ab>     and     <python .\office.py>


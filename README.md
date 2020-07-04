# Automatic Measurements of Fishes physical traits V2.


Automatic measurements of physical Traits of Fish.

Create a Python software to automatically measure fish's traits giving a photo.

There are two modes of operation. 

+ Give to the script the reference length. The name of the picture should contain the TL (Total Lenght in physical dimensions, cm, in, etc.) of the fish, with the tag "TL". With that number, the software computes the transformation from pixels to cm, inches, etc. If an error occurs, the distances will be shown in pixels. example of correctly formatted picture name: S schlegeli 15.7 SL-18.8 TL.jpg
+ Specify that there is reference tape in the picture.

The general parameters:

```
parser.add_argument("--path_pictures", help="path to pictures. return CNN feature detection")
parser.add_argument("--mode_ratio", help="in_name_TL, reference_tape")
parser.add_argument("--path_template", help="if mode_ratio=reference_tape, pass the path to the template")
parser.add_argument("--length", help="if mode_ratio=reference_tape, pass physical distante of tape")
parser.add_argument("--show_detection", help="return CNN feature detection")
parser.add_argument("--show_end_result", help="return image with dimension drawing")
parser.add_argument("--debug", help="show inline prints")
```

Relies on darkflow. Inference is configured to be done with CPU. (Python 3.7). To setup (Tested on ubunto 16):

## Scheme of procedure.
<img src="http://garisplace.com/img/amt_2_diagram.png" />

## All desired traits.
<img src="http://garisplace.com/img/fish_traits.jpeg" />



+ **Notebooks/exploratory.ipynb:** jupyter notebook with a demo.
+ **fish_detector.py:** isolated python script to measure fish's traits.
+ **Classes/:** Set of python classes to help fish_detector.py

** Still under development***



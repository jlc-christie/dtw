# dtw

[![Build Status](https://travis-ci.org/pierre-rouanet/dtw.svg?branch=master)](https://travis-ci.org/pierre-rouanet/dtw)

![An image of two small time series being matched from their warping paths](https://upload.wikimedia.org/wikipedia/commons/a/ab/Dynamic_time_warping.png "Dynamic Time Warping Example")

`dtw` is a Dynamic Time Warping (DTW) module in Python. DTW is a way of calculating
the similarity (or conversely, dissimilarity) between time series data. In brief,
it works by allowing the time domain to *warp* to a certain degree to minimise
the error. You can read more on DTW [here](https://en.wikipedia.org/wiki/Dynamic_time_warping).


## Examples (IPython Notebooks):

* [a simple example](./examples/simple%20example.ipynb)
* [a sound comparison based on DTW + MFCC](./examples/MFCC%20%2B%20DTW.ipynb)
* [simple speech recognition](./examples/speech-recognition.ipynb)


## Installation

```
pip install dtw
```

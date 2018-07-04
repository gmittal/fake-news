# Fake News Classification with Deep Learning
Kaggle's "Fake News" by the UTK Machine Learning Club. Achieves approximately 96.7% accuracy on Kaggle test data.

### Installation
Install dependencies with ```pip```.
```
pip install -r requirements.txt
```

### Usage
To train a new model, although a saved checkpoint is included with this repository, simply run the following.

```
python train.py
```

To try the model out, run:
```
python evaluate.py
```

### Model Architecture
<center>![model](save/model.png)</center>

### License
The MIT License (MIT)

Copyright (c) 2018 Gautam Mittal

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
